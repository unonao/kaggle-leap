import gc
import glob
import itertools
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
import xarray as xr
from adan import Adan
from google.cloud import storage
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import r2_score
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import utils
import wandb
from utils.humidity import cal_specific2relative_coef
from utils.metric import score


def get_valid_name(cfg):
    return f"{cfg.exp.valid_start[0]:02d}-{cfg.exp.valid_start[1]:02d}_{cfg.exp.valid_end[0]:02d}-{cfg.exp.valid_end[1]:02d}_{cfg.exp.valid_data_skip_mod}"


# physical constatns from (E3SM_ROOT/share/util/shr_const_mod.F90)
grav = 9.80616  # acceleration of gravity ~ m/s^2
cp = 1.00464e3  # specific heat of dry air   ~ J/kg/K
lv = 2.501e6  # latent heat of evaporation ~ J/kg
lf = 3.337e5  # latent heat of fusion      ~ J/kg
ls = lv + lf  # latent heat of sublimation ~ J/kg
rho_air = 101325.0 / (6.02214e26 * 1.38065e-23 / 28.966) / 273.15
rho_h20 = 1.0e3  # de


def rolling_mean(data, window_size):
    return uniform_filter1d(data, size=window_size, axis=1, mode="nearest")


def rolling_mean_std(data, window_size):
    # Rolling mean
    mean = uniform_filter1d(data, size=window_size, axis=1, mode="nearest")

    # Rolling std
    mean_sq = uniform_filter1d(data**2, size=window_size, axis=1, mode="nearest")
    variance = mean_sq - mean**2

    # Replace negative variance with zero
    variance[variance < 0] = 0
    std = np.sqrt(variance)
    return mean, std


class Scaler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.eps = cfg.exp.eps
        self.feat_mean_dict = pickle.load(
            open(
                Path(cfg.exp.scale_dir) / "x_mean_feat_dict.pkl",
                "rb",
            )
        )
        self.feat_std_dict = pickle.load(
            open(
                Path(cfg.exp.scale_dir) / "x_std_feat_dict.pkl",
                "rb",
            )
        )
        self.feat_mean_dict["cloud_snow_rate"] = 0.0
        self.feat_std_dict["cloud_snow_rate"] = 1.0
        self.y_mean = np.load(Path(cfg.exp.scale_dir) / "y_nanmean.npy")
        self.y_rms_sub = np.maximum(
            np.load(Path(cfg.exp.scale_dir) / "y_rms_sub.npy"),
            self.eps,
        )

        grid_path = "/kaggle/working/misc/grid_info/ClimSim_low-res_grid-info.nc"
        grid_info = xr.open_dataset(grid_path)
        self.hyai = grid_info["hyai"].to_numpy()
        self.hybi = grid_info["hybi"].to_numpy()
        self.hyam = grid_info["hyam"].to_numpy()
        self.hybm = grid_info["hybm"].to_numpy()

        self.tmelt_array = np.load(
            Path(cfg.exp.tmelt_tice_dir) / "tmelt_array.npy", allow_pickle=True
        )
        self.tice_array = np.load(
            Path(cfg.exp.tmelt_tice_dir) / "tice_array.npy",
            allow_pickle=True,
        )

        self.fill_target_index = [
            cfg.cols.col_names.index(col) for col in cfg.exp.fill_target
        ]

    def scale_input_1d(self, x):
        x = x[None, :]
        x, x_cat = self.scale_input(x)
        x, x_cat = x.reshape(-1), x_cat.reshape(-1)
        return x, x_cat

    def process_features(self, x_array):
        ps = x_array[:, 360]
        pressures_array = self.hyai * 1e5 + self.hybi[None, :] * ps[:, None]
        pressures_array = np.diff(pressures_array, n=1)

        feats = [x_array[:, :556]]
        if "relative_humidity_all" in self.cfg.exp.seq_feats:
            x_rh = (
                cal_specific2relative_coef(
                    temperature_array=x_array[:, 0:60],
                    near_surface_air_pressure=x_array[:, 360],
                    hyam=self.hyam,
                    hybm=self.hybm,
                    method=self.cfg.exp.rh_method,
                )
                * x_array[:, 60:120]
            )
            feats.append(x_rh)
        if "cloud_snow_rate" in self.cfg.exp.seq_feats:
            cloud_snow_rate_array = (
                np.clip(
                    x_array[:, 180:240]
                    / (x_array[:, 120:180] + x_array[:, 180:240] + self.eps),
                    0,
                    1,
                )
                - 0.5
            )
            feats.append(cloud_snow_rate_array)
        if "cloud_water" in self.cfg.exp.seq_feats:
            cloud_water_array = x_array[:, 120:180] + x_array[:, 180:240]
            feats.append(cloud_water_array)
        if "pressures" in self.cfg.exp.seq_feats:
            feats.append(pressures_array)
        if "pressures_all" in self.cfg.exp.seq_feats:
            feats.append(pressures_array)
        if "water" in self.cfg.exp.seq_feats:
            water_array = x_array[:, 60:120] + x_array[:, 120:180] + x_array[:, 180:240]
            feats.append(water_array)

        if "q2q3_mean" in self.cfg.exp.seq_feats:
            q2q3_mean_array = (x_array[:, 120:180] + x_array[:, 180:240]) / 2
            feats.append(q2q3_mean_array)
        if "uv_mean" in self.cfg.exp.seq_feats:
            uv_mean_array = (x_array[:, 240:300] + x_array[:, 300:360]) / 2
            feats.append(uv_mean_array)
        if "pbuf_mean" in self.cfg.exp.seq_feats:
            pbuf_mean_array = (
                x_array[:, 376 : 376 + 60]
                + x_array[:, 376 + 60 : 376 + 120]
                + x_array[:, 376 + 120 : 376 + 180]
            ) / 3
            feats.append(pbuf_mean_array)
        if "t_diff" in self.cfg.exp.seq_feats:
            t_diff_array = np.diff(
                x_array[:, 0:60], axis=1, append=0
            )  # 地上に近い方からの温度差を入れる
            feats.append(t_diff_array)
        if "q1_diff" in self.cfg.exp.seq_feats:
            q1_diff_array = np.diff(x_array[:, 60:120], axis=1, append=0)
            feats.append(q1_diff_array)
        if "q2_diff" in self.cfg.exp.seq_feats:
            q2_diff_array = np.diff(x_array[:, 120:180], axis=1, append=0)
            feats.append(q2_diff_array)
        if "q3_diff" in self.cfg.exp.seq_feats:
            q3_diff_array = np.diff(x_array[:, 180:240], axis=1, append=0)
            feats.append(q3_diff_array)
        if "u_diff" in self.cfg.exp.seq_feats:
            u_diff_array = np.diff(x_array[:, 240:300], axis=1, append=0)
            feats.append(u_diff_array)
        if "v_diff" in self.cfg.exp.seq_feats:
            v_diff_array = np.diff(x_array[:, 300:360], axis=1, append=0)
            feats.append(v_diff_array)
        if "ozone_diff" in self.cfg.exp.seq_feats:
            ozone_diff_array = np.diff(x_array[:, 376:436], axis=1, append=0)
            feats.append(ozone_diff_array)
        if "ch4_diff" in self.cfg.exp.seq_feats:
            ch4_diff_array = np.diff(x_array[:, 436:496], axis=1, append=0)
            feats.append(ch4_diff_array)
        if "n2o_diff" in self.cfg.exp.seq_feats:
            n2o_diff_array = np.diff(x_array[:, 496:556], axis=1, append=0)
            feats.append(n2o_diff_array)
        if "q2q3_mean_diff" in self.cfg.exp.seq_feats:
            q2q3_mean_diff_array = np.diff(q2q3_mean_array, axis=1, append=0)
            feats.append(q2q3_mean_diff_array)
        if "uv_mean_diff" in self.cfg.exp.seq_feats:
            uv_mean_diff_array = np.diff(uv_mean_array, axis=1, append=0)
            feats.append(uv_mean_diff_array)
        if "pbuf_mean_diff" in self.cfg.exp.seq_feats:
            pbuf_mean_diff_array = np.diff(pbuf_mean_array, axis=1, append=0)
            feats.append(pbuf_mean_diff_array)

        eps = 1e-60
        if "t_per_change" in self.cfg.exp.seq_feats:
            t_per_change_array = np.diff(x_array[:, 0:60], axis=1, append=0) / (
                x_array[:, 0:60] + eps
            )
            feats.append(t_per_change_array)
        if "q1_per_change" in self.cfg.exp.seq_feats:
            q1_per_change_array = np.diff(x_array[:, 60:120], axis=1, append=0) / (
                x_array[:, 60:120] + eps
            )
            feats.append(q1_per_change_array)
        if "q2_per_change" in self.cfg.exp.seq_feats:
            q2_per_change_array = np.diff(x_array[:, 120:180], axis=1, append=0) / (
                x_array[:, 120:180] + eps
            )
            feats.append(q2_per_change_array)
        if "q3_per_change" in self.cfg.exp.seq_feats:
            q3_per_change_array = np.diff(x_array[:, 180:240], axis=1, append=0) / (
                x_array[:, 180:240] + eps
            )
            feats.append(q3_per_change_array)
        if "u_per_change" in self.cfg.exp.seq_feats:
            u_per_change_array = np.diff(x_array[:, 240:300], axis=1, append=0) / (
                x_array[:, 240:300] + eps
            )
            feats.append(u_per_change_array)
        if "v_per_change" in self.cfg.exp.seq_feats:
            v_per_change_array = np.diff(x_array[:, 300:360], axis=1, append=0) / (
                x_array[:, 300:360] + eps
            )
            feats.append(v_per_change_array)
        if "ozone_per_change" in self.cfg.exp.seq_feats:
            ozone_per_change_array = np.diff(x_array[:, 376:436], axis=1, append=0) / (
                x_array[:, 376:436] + eps
            )
            feats.append(ozone_per_change_array)
        if "ch4_per_change" in self.cfg.exp.seq_feats:
            ch4_per_change_array = np.diff(x_array[:, 436:496], axis=1, append=0) / (
                x_array[:, 436:496] + eps
            )
            feats.append(ch4_per_change_array)
        if "n2o_per_change" in self.cfg.exp.seq_feats:
            n2o_per_change_array = np.diff(x_array[:, 496:556], axis=1, append=0) / (
                x_array[:, 496:556] + eps
            )
            feats.append(n2o_per_change_array)
        if "q2q3_mean_per_change" in self.cfg.exp.seq_feats:
            q2q3_mean_per_change_array = np.diff(q2q3_mean_array, axis=1, append=0) / (
                q2q3_mean_array + eps
            )
            feats.append(q2q3_mean_per_change_array)
        if "uv_mean_per_change" in self.cfg.exp.seq_feats:
            uv_mean_per_change_array = np.diff(uv_mean_array, axis=1, append=0) / (
                uv_mean_array + eps
            )
            feats.append(uv_mean_per_change_array)
        if "pbuf_mean_per_change" in self.cfg.exp.seq_feats:
            pbuf_mean_per_change_array = np.diff(pbuf_mean_array, axis=1, append=0) / (
                pbuf_mean_array + eps
            )
            feats.append(pbuf_mean_per_change_array)

        # 上との差分
        if "t_diff_pre" in self.cfg.exp.seq_feats:
            t_diff_pre_array = np.diff(x_array[:, 0:60], axis=1, prepend=0)
            feats.append(t_diff_pre_array)
        if "q1_diff_pre" in self.cfg.exp.seq_feats:
            q1_diff_pre_array = np.diff(x_array[:, 60:120], axis=1, prepend=0)
            feats.append(q1_diff_pre_array)
        if "q2_diff_pre" in self.cfg.exp.seq_feats:
            q2_diff_pre_array = np.diff(x_array[:, 120:180], axis=1, prepend=0)
            feats.append(q2_diff_pre_array)
        if "q3_diff_pre" in self.cfg.exp.seq_feats:
            q3_diff_pre_array = np.diff(x_array[:, 180:240], axis=1, prepend=0)
            feats.append(q3_diff_pre_array)
        if "u_diff_pre" in self.cfg.exp.seq_feats:
            u_diff_pre_array = np.diff(x_array[:, 240:300], axis=1, prepend=0)
            feats.append(u_diff_pre_array)
        if "v_diff_pre" in self.cfg.exp.seq_feats:
            v_diff_pre_array = np.diff(x_array[:, 300:360], axis=1, prepend=0)
            feats.append(v_diff_pre_array)
        if "ozone_diff_pre" in self.cfg.exp.seq_feats:
            ozone_diff_pre_array = np.diff(x_array[:, 376:436], axis=1, prepend=0)
            feats.append(ozone_diff_pre_array)
        if "ch4_diff_pre" in self.cfg.exp.seq_feats:
            ch4_diff_pre_array = np.diff(x_array[:, 436:496], axis=1, prepend=0)
            feats.append(ch4_diff_pre_array)
        if "n2o_diff_pre" in self.cfg.exp.seq_feats:
            n2o_diff_pre_array = np.diff(x_array[:, 496:556], axis=1, prepend=0)
            feats.append(n2o_diff_pre_array)
        if "q2q3_mean_diff_pre" in self.cfg.exp.seq_feats:
            q2q3_mean_diff_pre_array = np.diff(q2q3_mean_array, axis=1, prepend=0)
            feats.append(q2q3_mean_diff_pre_array)
        if "uv_mean_diff_pre" in self.cfg.exp.seq_feats:
            uv_mean_diff_pre_array = np.diff(uv_mean_array, axis=1, prepend=0)
            feats.append(uv_mean_diff_pre_array)
        if "pbuf_mean_diff_pre" in self.cfg.exp.seq_feats:
            pbuf_mean_diff_pre_array = np.diff(pbuf_mean_array, axis=1, prepend=0)
            feats.append(pbuf_mean_diff_pre_array)

        # scalar
        if "sum_energy" in self.cfg.exp.scalar_feats:
            sum_energy_array = x_array[:, [361, 362, 363, 371]].sum(axis=1)
            feats.append(sum_energy_array.reshape(-1, 1))
        if "sum_flux" in self.cfg.exp.scalar_feats:
            sum_flux_array = x_array[:, [362, 363, 371]].sum(axis=1)
            feats.append(sum_flux_array.reshape(-1, 1))
        if "energy_diff" in self.cfg.exp.scalar_feats:
            energy_diff_array = x_array[:, 361] - sum_flux_array
            feats.append(energy_diff_array.reshape(-1, 1))
        if "bowen_ratio" in self.cfg.exp.scalar_feats:
            bowen_ratio_array = x_array[:, 362] / x_array[:, 363]
            feats.append(bowen_ratio_array.reshape(-1, 1))
        if "sum_surface_stress" in self.cfg.exp.scalar_feats:
            sum_surface_stress_array = x_array[:, [364, 365]].sum(axis=1)
            feats.append(sum_surface_stress_array.reshape(-1, 1))
        if "net_radiative_flux" in self.cfg.exp.scalar_feats:
            net_radiative_flux_array = (
                x_array[:, 361] * x_array[:, 366] - x_array[:, 371]
            )
            feats.append(net_radiative_flux_array.reshape(-1, 1))
        if "global_solar_irradiance" in self.cfg.exp.scalar_feats:
            global_solar_irradiance_array = (
                x_array[:, 361] * (1 - x_array[:, 369]) * (1 - x_array[:, 370])
            )
            feats.append(global_solar_irradiance_array.reshape(-1, 1))
        if "global_longwave_flux" in self.cfg.exp.scalar_feats:
            global_longwave_flux_array = (
                x_array[:, 371] * (1 - x_array[:, 367]) * (1 - x_array[:, 368])
            )
            feats.append(global_longwave_flux_array.reshape(-1, 1))
        numerical_features = np.concatenate(feats, axis=1)

        # カテゴリ変数に変換
        q2_zeros = (x_array[:, 120:180] == 0).astype(np.int8)
        q3_zeros = (x_array[:, 180:240] == 0).astype(np.int8) + 2
        categorical_features = np.concatenate([q2_zeros, q3_zeros], axis=1)
        return numerical_features, categorical_features

    def scale_input(self, x):
        """
        prepare
        """
        x, x_cat = self.process_features(x)

        """
        scale
        """
        x[:, 0:60] = (x[:, 0:60] - self.feat_mean_dict["t_all"]) / (
            self.feat_std_dict["t_all"] + self.eps
        )
        x[:, 60:120] = (
            np.log1p(x[:, 60:120] * 1e9) - self.feat_mean_dict["q1_log_all"]
        ) / (self.feat_std_dict["q1_log_all"] + self.eps)
        x[:, 120:180] = (
            np.log1p(x[:, 120:180] * 1e9) - self.feat_mean_dict["q2_log_all"]
        ) / (self.feat_std_dict["q2_log_all"] + self.eps)
        x[:, 180:240] = (
            np.log1p(x[:, 180:240] * 1e9) - self.feat_mean_dict["q3_log_all"]
        ) / (self.feat_std_dict["q3_log_all"] + self.eps)
        x[:, 240:300] = (x[:, 240:300] - self.feat_mean_dict["u_all"]) / (
            self.feat_std_dict["u_all"] + self.eps
        )
        x[:, 300:360] = (x[:, 300:360] - self.feat_mean_dict["v_all"]) / (
            self.feat_std_dict["v_all"] + self.eps
        )
        x[:, 360:376] = (
            x[:, 360:376] - self.feat_mean_dict["base"][360:376]
        ) / self.feat_std_dict["base"][360:376]
        x[:, 376:436] = (x[:, 376:436] - self.feat_mean_dict["ozone_all"]) / (
            self.feat_std_dict["ozone_all"] + self.eps
        )
        x[:, 436:496] = (x[:, 436:496] - self.feat_mean_dict["ch4_all"]) / (
            self.feat_std_dict["ch4_all"] + self.eps
        )
        x[:, 496:556] = (x[:, 496:556] - self.feat_mean_dict["n2o_all"]) / (
            self.feat_std_dict["n2o_all"] + self.eps
        )

        for i, key in enumerate(self.cfg.exp.seq_feats):
            start = 556 + i * 60
            end = 556 + (i + 1) * 60
            x[:, start:end] = (x[:, start:end] - self.feat_mean_dict[key]) / np.maximum(
                self.feat_std_dict[key], self.eps
            )
        for i, key in enumerate(self.cfg.exp.scalar_feats):
            start = 556 + len(self.cfg.exp.seq_feats) * 60 + i
            end = 556 + len(self.cfg.exp.seq_feats) * 60 + i + 1
            x[:, start:end] = (x[:, start:end] - self.feat_mean_dict[key]) / np.maximum(
                self.feat_std_dict[key], self.eps
            )

        # outlier_std_rate を超えたらclip
        return np.clip(
            x,
            -self.cfg.exp.outlier_std_rate,
            self.cfg.exp.outlier_std_rate,
        ), x_cat

    """
    def scale_output_1d(self, y):
        y = y[None, :]
        y = self.scale_output(y).reshape(-1)
        return y
    """

    def scale_output(self, y):
        y = (y - self.y_mean) / self.y_rms_sub
        return y

    def inv_scale_output(self, y, original_x, post_process=True):
        y = y * self.y_rms_sub + self.y_mean
        for i in range(self.y_rms_sub.shape[0]):
            if self.y_rms_sub[i] < self.eps * 1.1:
                y[:, i] = self.y_mean[i]

        if post_process:
            y = self.post_process(y, original_x)

        return y

    def post_process(self, y, original_x):
        # tmelt以上の値を置き換える
        tmelt_cond = original_x[:, :60] > self.tmelt_array
        y[:, 180:240] = np.where(
            tmelt_cond, original_x[:, 180:240] / (-1200), y[:, 180:240]
        )
        # tice以下の値を置き換える
        tmelt_cond = original_x[:, :60] < self.tice_array
        y[:, 120:180] = np.where(
            tmelt_cond, original_x[:, 120:180] / (-1200), y[:, 120:180]
        )

        # fill target はすべて置き換える
        y[:, self.fill_target_index] = original_x[:, self.fill_target_index] * (
            -1 / 1200
        )
        return y

    def filter_and_scale(self, x, y):
        filter_bool = np.all(
            y <= 1e60, axis=1
        )  # y が lower_bound と upper_bound の間に収まらなければその行をスキップしていた

        x, x_cat = self.scale_input(x)
        y = self.scale_output(y)
        return x, x_cat, y, filter_bool

    def filter_and_scale_1d(self, x, y):
        x, x_cat = self.scale_input_1d(x)
        y = self.scale_output(y)
        return x, x_cat, y


class LeapLightningDataModule(LightningDataModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.scaler = Scaler(cfg)
        self.cfg = cfg
        self.rng = random.Random(self.cfg.exp.seed)
        self.train_dataset = self._make_dataset("train")
        # self.valid_dataset = self._make_dataset("valid")
        grid_path = "/kaggle/working/misc/grid_info/ClimSim_low-res_grid-info.nc"
        grid_info = xr.open_dataset(grid_path)
        self.hyai = grid_info["hyai"].to_numpy()
        self.hybi = grid_info["hybi"].to_numpy()

    class Val2Dataset(Dataset):
        def __init__(self, cfg, valid_df, scaler, hyai, hybi):
            self.cfg = cfg
            self.scaler = scaler
            self.hyai = hyai
            self.hybi = hybi
            # 提供データは cam_in_SNOWHICE は削除済みなので削除しないが、idを削除する
            self.x = valid_df[:, 1:557].to_numpy()
            self.y = valid_df[:, 557:].to_numpy()

            # 長さは384の倍数にする
            mod = self.x.shape[0] % 384
            if mod > 0:
                self.x = self.x[:-mod]
                self.y = self.y[:-mod]

        def __len__(self):
            return self.x.shape[0] // 384

        def __getitem__(self, index):
            original_x = self.x[index * 384 : (index + 1) * 384]
            original_y = self.y[index * 384 : (index + 1) * 384]
            x, x_cat, y, filter_bool = self.scaler.filter_and_scale(
                original_x, original_y
            )
            return (
                torch.from_numpy(x),
                torch.from_numpy(x_cat),
                torch.from_numpy(y),
                torch.from_numpy(filter_bool),
                torch.from_numpy(original_x),
                torch.from_numpy(original_y),
            )

    class TestDataset(Dataset):
        def __init__(self, cfg, test_df, scaler, hyai, hybi):
            self.cfg = cfg
            self.scaler = scaler
            self.hyai = hyai
            self.hybi = hybi
            # 提供データは cam_in_SNOWHICE は削除済みなので削除しないが、idを削除する
            self.x = test_df[:, 1:].to_numpy()

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, index):
            original_x = self.x[index]
            x, x_cat = self.scaler.scale_input_1d(original_x)
            return (
                torch.from_numpy(x),
                torch.from_numpy(x_cat),
                torch.from_numpy(original_x),
            )

    def train_dataloader(self):
        return (
            wds.WebLoader(
                self.train_dataset,
                batch_size=None,
                num_workers=self.cfg.exp.num_workers,
            )
            .shuffle(7)
            .batched(
                batchsize=self.cfg.exp.train_batch_size,
                partial=False,
            )
        )

    def val_dataloader(self):
        dataloader, _ = self.val2_dataloader()
        return dataloader

    def val2_dataloader(self):
        """
        validation合わせるために作った推論用のdataloader
        """
        self.valid_df = pl.read_parquet(
            self.cfg.exp.valid_path,
            n_rows=(None if self.cfg.debug is False else 384 * 2),
        )
        self.val2_dataset = self.Val2Dataset(
            self.cfg, self.valid_df, self.scaler, self.hyai, self.hybi
        )
        return DataLoader(
            self.val2_dataset,
            batch_size=self.cfg.exp.valid_batch_size,
            num_workers=self.cfg.exp.num_workers,
            shuffle=False,
            pin_memory=False,
        ), self.valid_df

    def test_dataloader(self):
        self.test_df = pl.read_parquet(
            self.cfg.exp.test_path, n_rows=(None if self.cfg.debug is False else 500)
        )
        self.test_dataset = self.TestDataset(
            self.cfg, self.test_df, self.scaler, self.hyai, self.hybi
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.exp.valid_batch_size * 128,
            num_workers=self.cfg.exp.num_workers,
            shuffle=False,
            pin_memory=False,
        ), self.test_df

    def _make_tar_list(self, mode="train"):
        tar_list = []

        start_year, start_month = (
            self.cfg.exp.train_start if mode == "train" else self.cfg.exp.valid_start
        )
        end_year, end_month = (
            self.cfg.exp.train_end if mode == "train" else self.cfg.exp.valid_end
        )
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                if (year == start_year and month < start_month) or (
                    year == end_year and month > end_month
                ):
                    continue
                tmp = sorted(
                    glob.glob(
                        f"{self.cfg.exp.dataset_dir}/shards_{year:04d}-{month:02d}/*.tar"
                    )
                )
                tar_list += tmp
        # 1/data_skip_mod の数にする
        if mode == "train" and self.cfg.exp.train_data_skip_mod:
            tar_list = tar_list[:: self.cfg.exp.train_data_skip_mod]
        elif mode == "valid" and self.cfg.exp.valid_data_skip_mod:
            tar_list = tar_list[:: self.cfg.exp.valid_data_skip_mod]

        if mode == "train" and self.cfg.exp.additional_start is not None:
            tmp_tar_list = []
            start_year, start_month = self.cfg.exp.additional_start
            end_year, end_month = self.cfg.exp.additional_end
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    if (year == start_year and month < start_month) or (
                        year == end_year and month > end_month
                    ):
                        continue
                    tmp = sorted(
                        glob.glob(
                            f"{self.cfg.exp.dataset_dir}/shards_{year:04d}-{month:02d}/*.tar"
                        )
                    )
                    tmp_tar_list += tmp
            if self.cfg.exp.additional_data_skip_mod:
                tar_list += tmp_tar_list[:: self.cfg.exp.additional_data_skip_mod]

        print(mode, f"{len(tar_list)=}", tar_list[-1])
        return tar_list

    def _get_dataset_size(self, mode="train"):
        start_year, start_month = (
            self.cfg.exp.train_start if mode == "train" else self.cfg.exp.valid_start
        )
        end_year, end_month = (
            self.cfg.exp.train_end if mode == "train" else self.cfg.exp.valid_end
        )
        total_size = 0
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                if (year == start_year and month < start_month) or (
                    year == end_year and month > end_month
                ):
                    continue
                tmp = 0
                paths = glob.glob(
                    f"{self.cfg.exp.dataset_dir}/shards_{year:04d}-{month:02d}/dataset-size.json"
                )
                for path in paths:
                    with open(path, "r") as f:
                        json_data = json.load(f)
                        dataset_size = json_data["dataset size"]
                        tmp += dataset_size
                total_size += tmp
        # 1/data_skip_mod の数にする
        if mode == "train" and self.cfg.exp.train_data_skip_mod:
            total_size = total_size // self.cfg.exp.train_data_skip_mod
        elif mode == "valid" and self.cfg.exp.valid_data_skip_mod:
            total_size = total_size // self.cfg.exp.valid_data_skip_mod

        if mode == "train" and self.cfg.exp.additional_start is not None:
            tmp_total_size = 0
            start_year, start_month = self.cfg.exp.additional_start
            end_year, end_month = self.cfg.exp.additional_end
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    if (year == start_year and month < start_month) or (
                        year == end_year and month > end_month
                    ):
                        continue
                    tmp = 0
                    paths = glob.glob(
                        f"{self.cfg.exp.dataset_dir}/shards_{year:04d}-{month:02d}/dataset-size.json"
                    )
                    for path in paths:
                        with open(path, "r") as f:
                            json_data = json.load(f)
                            dataset_size = json_data["dataset size"]
                            tmp += dataset_size
                    tmp_total_size += tmp
            if self.cfg.exp.additional_data_skip_mod:
                total_size += tmp_total_size // self.cfg.exp.additional_data_skip_mod

        return total_size

    def _make_dataset(self, mode="train"):
        tar_list = self._make_tar_list(mode)
        dataset_size = self._get_dataset_size(mode)
        file_size = (
            dataset_size // 384
        )  # 1ファイルに約384ずつまとめているのでそれで割っておく

        print(f"{mode=}", f"{dataset_size=}", f"{file_size=}")
        dataset = None
        if mode == "train":
            dataset = wds.WebDataset(urls=tar_list, shardshuffle=True).shuffle(
                100, rng=self.rng
            )
        else:
            dataset = wds.WebDataset(urls=tar_list, shardshuffle=False)

        dataset = (
            dataset.decode().to_tuple("input.npy", "output.npy").with_length(file_size)
        )

        def _process(source):
            for sample in source:
                original_x, original_y = sample
                original_x = np.delete(original_x, 375, 1)
                x, x_cat, y, mask = self.scaler.filter_and_scale(original_x, original_y)
                x, x_cat, y, mask, original_x, original_y = (
                    torch.tensor(x),
                    torch.tensor(x_cat),
                    torch.tensor(y),
                    torch.tensor(mask),
                    torch.tensor(original_x),
                    torch.tensor(original_y),
                )
                yield x, x_cat, y, mask, original_x, original_y

        dataset = dataset.compose(_process)
        return dataset


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2
    height and width size will be changed to size-4.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        kernel_size=(3, 1),
        padding=(1, 0),
        use_batch_norm=True,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_batch_norm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    mid_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    mid_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_batch_norm=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d((2, 1)),
            DoubleConv(in_channels, out_channels, use_batch_norm=use_batch_norm),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_batch_norm=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                in_channels // 2,
                use_batch_norm=use_batch_norm,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=(2, 1), stride=2
            )
            self.conv = DoubleConv(
                in_channels, out_channels, use_batch_norm=use_batch_norm
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BottleneckEncoder(nn.Sequential):
    # https://github.com/qubvel/segmentation_models.pytorch/blob/3bf4d6ef2bc9d41c2ab3436838aa22375dd0f23a/segmentation_models_pytorch/base/heads.py#L13
    def __init__(self, in_channels, out_nums, pooling="avg", dropout=0.2):
        if pooling not in ("max", "avg"):
            raise ValueError(
                "Pooling should be one of ('max', 'avg'), got {}.".format(pooling)
            )
        pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, out_nums, bias=True)
        super().__init__(pool, flatten, dropout, linear)


class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        bottleneck_out_nums,
        depth=4,
        dropout=0.2,
        n_base_channels=32,
        use_batch_norm=True,
    ):
        super(UNet, self).__init__()
        bilinear = False
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(
            n_channels,
            n_base_channels,
            use_batch_norm=use_batch_norm,
        )
        self.downs = nn.ModuleList(
            [
                Down(
                    n_base_channels * (2**i),
                    n_base_channels * (2 ** (i + 1)),
                    use_batch_norm,
                )
                for i in range(depth)
            ]
        )
        self.ups = nn.ModuleList(
            [
                Up(
                    n_base_channels * (2 ** (depth - i)),
                    n_base_channels * (2 ** (depth - i - 1)),
                    bilinear,
                    use_batch_norm,
                )
                for i in range(depth)
            ]
        )

        self.outc = OutConv(n_base_channels, n_classes)

        self.bottleneck_encoder = BottleneckEncoder(
            n_base_channels * (2**depth),
            out_nums=bottleneck_out_nums,
            pooling="avg",
            dropout=dropout,
        )

    def forward(self, x):
        x1 = self.inc(x)
        xs = [x1]
        for down in self.downs:
            xs.append(down(xs[-1]))
        x = xs[-1]
        for i, up in enumerate(self.ups):
            x = up(x, xs[-2 - i])
        logits = self.outc(x)
        bottleneck_feat = self.bottleneck_encoder(xs[-1])
        return logits, bottleneck_feat


class MLP(nn.Module):
    def __init__(self, in_size, hidden_sizes, use_layer_norm=False):
        super(MLP, self).__init__()
        layers = []
        previous_size = in_size
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_size, hidden_size))
            if i != len(hidden_sizes) - 1:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_size))
                layers.append(nn.LeakyReLU(inplace=True))
            previous_size = hidden_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class LeapModel(nn.Module):
    def __init__(
        self,
        same_height_hidden_sizes=[60, 60],
        output_hidden_sizes=[60, 60],
        use_input_layer_norm=False,
        use_output_layer_norm=True,
        use_batch_norm=True,
        embedding_dim=5,
        categorical_embedding_dim=5,
        depth=4,
        n_unet=2,
        dropout=0.2,
        n_base_channels=32,
        seq_feats=[],
        scalar_feats=[],
    ):
        super().__init__()
        self.seq_feats = seq_feats
        self.scalar_feats = scalar_feats
        num_embeddings = 60
        self.positional_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.constant_encoder = nn.Linear(1, 60)
        self.categorical_embedding = nn.Embedding(240, categorical_embedding_dim)

        previous_size = 9 + 16 + embedding_dim
        previous_size += len(seq_feats)
        previous_size += len(scalar_feats)
        previous_size += 2 * categorical_embedding_dim
        self.same_height_encoder = MLP(
            previous_size, same_height_hidden_sizes, use_layer_norm=use_input_layer_norm
        )

        unet_layers = []
        in_channels = same_height_hidden_sizes[-1]
        for _ in range(n_unet):
            unet_layers.append(
                UNet(
                    n_channels=in_channels,
                    n_classes=n_base_channels,
                    bottleneck_out_nums=8,
                    depth=depth,
                    dropout=dropout,
                    n_base_channels=n_base_channels,
                    use_batch_norm=use_batch_norm,
                )
            )
            in_channels = n_base_channels
        self.unet_layers = nn.ModuleList(unet_layers)

        self.t_head = MLP(
            in_channels,
            output_hidden_sizes + [2],
            use_layer_norm=use_output_layer_norm,
        )
        self.q1_head = MLP(
            in_channels,
            output_hidden_sizes + [2],
            use_layer_norm=use_output_layer_norm,
        )
        self.cloud_water_head = MLP(
            in_channels,
            output_hidden_sizes + [4],
            use_layer_norm=use_output_layer_norm,
        )
        self.wind_head = nn.ModuleList(
            [
                MLP(
                    in_channels,
                    output_hidden_sizes + [2],
                    use_layer_norm=use_output_layer_norm,
                )
                for _ in range(2)
            ]
        )

    def forward(self, x, x_cat):
        x_state_t = x[:, :60].unsqueeze(-1)
        x_state_q0001 = x[:, 60:120].unsqueeze(-1)
        x_state_q0002 = x[:, 120:180].unsqueeze(-1)
        x_state_q0003 = x[:, 180:240].unsqueeze(-1)
        x_state_u = x[:, 240:300].unsqueeze(-1)
        x_state_v = x[:, 300:360].unsqueeze(-1)
        x_constant_16 = x[:, 360:376].unsqueeze(-1)
        x_constant_16 = self.constant_encoder(x_constant_16).transpose(1, 2)
        x_pbuf_ozone = x[:, 376:436].unsqueeze(-1)
        x_pbuf_CH4 = x[:, 436:496].unsqueeze(-1)
        x_pbuf_N2O = x[:, 496:556].unsqueeze(-1)
        x_position = self.positional_embedding(
            torch.LongTensor(range(60)).repeat(x.shape[0], 1).to(x.device)
        )
        input_list = [
            x_state_t,
            x_state_q0001,
            x_state_q0002,
            x_state_q0003,
            x_state_u,
            x_state_v,
            x_constant_16,
            x_pbuf_ozone,
            x_pbuf_CH4,
            x_pbuf_N2O,
            x_position,
        ]
        for i, _ in enumerate(self.seq_feats):
            start = 556 + i * 60
            end = 556 + (i + 1) * 60
            x_seq = x[:, start:end].unsqueeze(-1)
            input_list.append(x_seq)
        for i, _ in enumerate(self.scalar_feats):
            start = 556 + len(self.seq_feats) * 60 + i
            end = 556 + len(self.seq_feats) * 60 + i + 1
            x_scalar = x[:, start:end].unsqueeze(-1)
            x_scalar = self.constant_encoder(x_scalar).transpose(1, 2)
            input_list.append(x_scalar)

        # (batch, 120) -> (batch, 120, 5)
        x_cat = self.categorical_embedding(x_cat)
        # (batch, 120, 5) -> (batch, 60, 10)
        x_cat = torch.cat(
            [
                x_cat[:, :60, :],
                x_cat[:, 60:120, :],
            ],
            dim=2,
        )
        input_list.append(x_cat)

        #  (batch, 60, dim)
        x = torch.cat(
            input_list,
            dim=2,
        )

        # (batch, 60, dim) -> (batch, 60, same_height_hidden_sizes[-1]*n_feat_channels)
        x = self.same_height_encoder(x)
        x = x.transpose(-1, -2).unsqueeze(-1)

        # input: (batch, n_base_channels, 60, 1)
        h = x
        for unet_layer in self.unet_layers:
            x_pre = h
            h, class_logits = unet_layer(x_pre)

        x = x_pre - h

        x = x.squeeze(-1)  # ->(batch, n_base_channels, 60)
        x = x.transpose(-1, -2)  # ->(batch, 60, n_base_channels)

        out_t = self.t_head(x)  # -> (batch, 60, 1)
        out_t = out_t[:, :, 0:1].exp() - out_t[:, :, 1:2].exp()

        out_q1 = self.q1_head(x)
        out_q1 = out_q1[:, :, 0:1].exp() - out_q1[:, :, 1:2].exp()

        out_cw = self.cloud_water_head(x)
        out_q2 = out_cw[:, :, 0:1].exp() - out_cw[:, :, 1:2].exp()
        out_q3 = out_cw[:, :, 2:3].exp() - out_cw[:, :, 3:4].exp()

        out_u = self.wind_head[0](x)
        out_u = out_u[:, :, 0:1].exp() - out_u[:, :, 1:2].exp()

        out_v = self.wind_head[1](x)
        out_v = out_v[:, :, 0:1].exp() - out_v[:, :, 1:2].exp()

        out = torch.cat([out_t, out_q1, out_q2, out_q3, out_u, out_v], dim=2)
        out = out.transpose(-1, -2)
        out = out.flatten(start_dim=1)
        out = torch.cat([out, class_logits], dim=1)

        return out


class LeapLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = LeapModel(
            **cfg.exp.model,
            seq_feats=cfg.exp.seq_feats,
            scalar_feats=cfg.exp.scalar_feats,
        )
        self.scaler = Scaler(cfg)
        self.loss_fc = nn.MSELoss()  # Using MSE for regression
        self.model_ema = None
        if self.cfg.exp.ema.use_ema:
            print("Using EMA")
            self.model_ema = ModelEmaV2(self.model, self.cfg.exp.ema.decay)

        self.valid_name = get_valid_name(cfg)
        self.torch_dtype = torch.float64 if "64" in cfg.exp.precision else torch.float32

        unuse_cols = list(itertools.chain.from_iterable(cfg.exp.unuse_cols_list))
        self.unuse_cols_index = [cfg.cols.col_names.index(col) for col in unuse_cols]
        self.use_cols_index = torch.tensor(
            [i for i in range(368) if i not in self.unuse_cols_index]
        )

        self.valid_preds = []
        self.valid_labels = []
        self.valid_original_xs = []

        ss_df = pl.read_csv(
            "input/leap-atmospheric-physics-ai-climsim/sample_submission.csv", n_rows=1
        )
        self.weight_array = ss_df.select(
            [x for x in ss_df.columns if x != "sample_id"]
        ).to_numpy()[0]

    def training_step(self, batch, batch_idx):
        mode = "train"
        x, x_cat, y, mask, _, _ = batch
        x, x_cat, y, mask = (
            torch.flatten(x, start_dim=0, end_dim=1),
            torch.flatten(x_cat, start_dim=0, end_dim=1),
            torch.flatten(y, start_dim=0, end_dim=1),
            torch.flatten(mask, start_dim=0, end_dim=1),
        )
        x, y = (
            x.to(self.torch_dtype),
            y.to(self.torch_dtype),
        )
        x_cat = x_cat.to(torch.long)
        out = self.__pred(x, x_cat, mode)

        out_masked = out[mask]
        y_masked = y[mask]
        loss = self.loss_fc(
            out_masked[:, self.use_cols_index], y_masked[:, self.use_cols_index]
        )
        self.log(
            f"{mode}_loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        mode = "valid"
        x, x_cat, y, mask, original_x, original_y = batch
        x, x_cat, y, mask, original_x, original_y = (
            torch.flatten(x, start_dim=0, end_dim=1),
            torch.flatten(x_cat, start_dim=0, end_dim=1),
            torch.flatten(y, start_dim=0, end_dim=1),
            torch.flatten(mask, start_dim=0, end_dim=1),
            torch.flatten(original_x, start_dim=0, end_dim=1),
            torch.flatten(original_y, start_dim=0, end_dim=1),
        )
        x, y = (
            x.to(self.torch_dtype),
            y.to(self.torch_dtype),
        )
        x_cat = x_cat.to(torch.long)
        out = self.__pred(x, x_cat, mode)
        out_masked = out[mask]
        y_masked = y[mask]
        loss = self.loss_fc(
            out_masked[:, self.use_cols_index], y_masked[:, self.use_cols_index]
        )
        self.log(
            f"{mode}_loss/{self.valid_name}",
            loss.detach().item(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.valid_preds.append(out.cpu().to(torch.float64).numpy())
        self.valid_labels.append(original_y.cpu().to(torch.float64).numpy())
        self.valid_original_xs.append(original_x.cpu().to(torch.float64).numpy())
        return loss

    def predict_step(self, batch, batch_idx):
        mode = "test"
        x, x_cat, _ = batch
        x = x.to(self.torch_dtype)
        x_cat = x_cat.to(torch.long)
        out = self.__pred(x, x_cat, mode)
        return out

    def __pred(self, x, x_cat, mode: str) -> torch.Tensor:
        if (mode == "valid" or mode == "test") and (self.model_ema is not None):
            out = self.model_ema.module(x, x_cat)
        else:
            out = self.model(x, x_cat)
        return out

    def on_after_backward(self):
        if self.model_ema is not None:
            self.model_ema.update(self.model)

    def on_validation_epoch_end(self):
        valid_preds = np.concatenate(self.valid_preds, axis=0).astype(np.float64)
        valid_labels = np.concatenate(self.valid_labels, axis=0).astype(np.float64)
        valid_original_xs = np.concatenate(self.valid_original_xs, axis=0).astype(
            np.float64
        )
        valid_preds = self.scaler.inv_scale_output(valid_preds, valid_original_xs)

        r2_scores = r2_score(
            valid_labels * self.weight_array,
            valid_preds * self.weight_array,
        )
        print(f"{r2_scores=}")
        self.log(
            f"valid_r2_score/{self.valid_name}",
            r2_scores,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.valid_preds = []
        self.valid_labels = []
        self.valid_original_xs = []
        gc.collect()

    def configure_optimizers(self):
        optimizer = None
        if self.cfg.exp.optimizer.name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.exp.optimizer.lr,
                weight_decay=self.cfg.exp.optimizer.weight_decay,
            )
        elif self.cfg.exp.optimizer.name == "RAdam":
            optimizer = torch.optim.RAdam(
                self.model.parameters(),
                lr=self.cfg.exp.optimizer.lr,
                weight_decay=self.cfg.exp.optimizer.weight_decay,
            )
        elif self.cfg.exp.optimizer.name == "Adan":
            optimizer = Adan(
                self.model.parameters(),
                lr=self.cfg.exp.optimizer.lr,
                weight_decay=self.cfg.exp.optimizer.weight_decay,
                betas=self.cfg.exp.optimizer.opt_betas,
                eps=self.cfg.exp.optimizer.eps,
                max_grad_norm=self.cfg.exp.optimizer.max_grad_norm,
                no_prox=self.cfg.exp.optimizer.no_prox,
            )

        if self.cfg.exp.scheduler.name == "CosineAnnealingWarmRestarts":
            # 1epoch分をwarmupとするための記述
            num_warmup_steps = (
                math.ceil(self.trainer.max_steps / self.cfg.exp.max_epochs) * 1
                if self.cfg.exp.scheduler.use_one_epoch_warmup
                else 0
            )
            if self.cfg.exp.val_check_interval:
                num_warmup_steps = min(num_warmup_steps, self.cfg.exp.val_check_warmup)
            print(f"{num_warmup_steps=}")
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=self.trainer.max_steps,
                num_warmup_steps=num_warmup_steps,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif self.cfg.exp.scheduler.name == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.cfg.exp.scheduler.base_lr,
                max_lr=self.cfg.exp.scheduler.max_lr,
                mode="triangular2",
                step_size_up=max(
                    int(
                        self.trainer.max_steps / self.cfg.exp.scheduler.num_cycles * 0.1
                    ),
                    1,
                ),
                step_size_down=int(
                    self.trainer.max_steps / self.cfg.exp.scheduler.num_cycles * 0.9
                ),
                cycle_momentum=False,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif self.cfg.exp.scheduler.name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.cfg.exp.scheduler.factor,
                patience=self.cfg.exp.scheduler.patience,
                threshold=self.cfg.exp.scheduler.threshold,
                threshold_mode=self.cfg.exp.scheduler.threshold_mode,
                cooldown=self.cfg.exp.scheduler.cooldown,
                min_lr=self.cfg.exp.scheduler.min_lr,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "frequency": self.cfg.exp.val_check_interval
                    if self.cfg.exp.val_check_interval
                    else 1,
                    "monitor": f"valid_r2_score/{self.valid_name}",
                    "interval": "step" if self.cfg.exp.val_check_interval else "epoch",
                },
            }


def train(cfg: DictConfig, output_path: Path, pl_logger) -> None:
    valid_name = get_valid_name(cfg)
    monitor = f"valid_r2_score/{valid_name}"
    dm = LeapLightningDataModule(cfg)
    model = LeapLightningModule(cfg)
    checkpoint_cb = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        verbose=True,
        monitor=monitor,
        mode="max",
        save_top_k=1,
        save_last=False,
        enable_version_counter=False,
    )
    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = RichProgressBar()  # leave=True
    model_summary = RichModelSummary(max_depth=2)
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=cfg.exp.early_stopping_patience,
        mode="max",
    )
    if cfg.debug:
        cfg.exp.max_epochs = 2

    trainer = Trainer(
        default_root_dir=output_path,
        accelerator=cfg.exp.accelerator,
        precision=cfg.exp.precision,
        max_epochs=cfg.exp.max_epochs,
        max_steps=cfg.exp.max_epochs
        * len(dm.train_dataset)
        // cfg.exp.train_batch_size
        // cfg.exp.accumulate_grad_batches,
        gradient_clip_val=cfg.exp.gradient_clip_val,
        accumulate_grad_batches=cfg.exp.accumulate_grad_batches,
        logger=pl_logger,
        log_every_n_steps=1,
        limit_train_batches=None if cfg.debug is False else 2,
        limit_val_batches=None if cfg.debug is False else 2,
        # deterministic=True,
        callbacks=[
            checkpoint_cb,
            lr_monitor,
            progress_bar,
            model_summary,
            early_stopping,
        ],
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        # sync_batchnorm=True,
        val_check_interval=cfg.exp.val_check_interval,
    )
    trainer.fit(model, dm, ckpt_path=cfg.exp.resume_ckpt_path)

    # copy checkpoint_cb.best_model_path
    shutil.copy(
        checkpoint_cb.best_model_path,
        output_path / "checkpoints" / "best_model.ckpt",
    )

    del model, dm, trainer
    gc.collect()
    torch.cuda.empty_cache()


"""
def predict_valid(cfg: DictConfig, output_path: Path) -> None:
    # TODO: チームを組むならvalidationデータセットを揃えて出力を保存する

    torch_dtype = torch.float64 if "64" in cfg.exp.precision else torch.float32

    valid_name = get_valid_name(cfg)
    checkpoint_path = (
        output_path / "checkpoints" / "best_model.ckpt"
        if cfg.exp.pred_checkpoint_path is None
        else cfg.exp.pred_checkpoint_path
    )
    model_module = LeapLightningModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    if cfg.exp.ema.use_ema:
        model_module.model = model_module.model_ema.module
    model = model_module.model

    dm = LeapLightningDataModule(cfg)
    dataloader = dm.val_dataloader()

    original_xs = []
    preds = []
    labels = []
    model = model.to("cuda")
    model.eval()
    for x, x_cat, _, _, original_x, original_y in tqdm(dataloader):
        x, x_cat, original_x, original_y = (
            x.to("cuda"),
            x_cat.to("cuda"),
            original_x.to("cuda"),
            original_y.to("cuda"),
        )
        with torch.no_grad():
            out = model(x.to(torch_dtype), x_cat.to(torch.long))

        # バッチごとに分割されているので結合
        out = np.concatenate(out.detach().cpu().numpy(), axis=0)
        original_x = np.concatenate(original_x.cpu().to(torch.float64).numpy(), axis=0)
        original_y = np.concatenate(original_y.cpu().to(torch.float64).numpy(), axis=0)
        preds.append(out)
        original_xs.append(original_x)
        labels.append(original_y)
        if cfg.debug:
            break

    with utils.trace("save predict"):
        original_xs = np.concatenate(original_xs, axis=0)
        preds = Scaler(cfg).inv_scale_output(np.concatenate(preds, axis=0), original_xs)
        labels = np.concatenate(labels, axis=0)

        original_xs_df = pd.DataFrame(
            original_xs, columns=[i for i in range(original_xs.shape[1])]
        ).reset_index()
        original_xs_df.to_parquet(output_path / "original_xs.parquet")
        del original_xs_df
        gc.collect()

        original_predict_df = pd.DataFrame(
            preds, columns=[i for i in range(preds.shape[1])]
        ).reset_index()
        original_predict_df.to_parquet(output_path / "predict.parquet")
        del original_predict_df
        gc.collect()

        original_label_df = pd.DataFrame(
            labels, columns=[i for i in range(labels.shape[1])]
        ).reset_index()
        original_label_df.to_parquet(output_path / "label.parquet")
        del original_label_df
        gc.collect()

    del original_xs
    gc.collect()

    # weight (weight zero もあるのでかけておく)
    ss_df = pl.read_csv(
        "input/leap-atmospheric-physics-ai-climsim/sample_submission.csv", n_rows=1
    )
    weight_array = ss_df.select(
        [x for x in ss_df.columns if x != "sample_id"]
    ).to_numpy()[0]
    predict_df = pd.DataFrame(
        preds * weight_array, columns=[i for i in range(preds.shape[1])]
    ).reset_index()
    label_df = pd.DataFrame(
        labels * weight_array, columns=[i for i in range(labels.shape[1])]
    ).reset_index()

    r2_scores = score(label_df, predict_df, "index", multioutput="raw_values")
    r2_score_dict = {
        col: r2 for col, r2 in dict(zip(cfg.cols.col_names, r2_scores)).items()
    }
    pickle.dump(r2_score_dict, open(output_path / "r2_score_dict.pkl", "wb"))

    r2_score = float(np.array([v for v in r2_score_dict.values()]).mean())
    print(f"{r2_score=}")

    wandb.log(
        {
            f"r2_score/{valid_name}": r2_score,
        }
    )
    del predict_df, label_df
    gc.collect()
"""


def predict_val2(cfg: DictConfig, output_path: Path) -> None:
    torch_dtype = torch.float64 if "64" in cfg.exp.precision else torch.float32
    checkpoint_path = (
        output_path / "checkpoints" / "best_model.ckpt"
        if cfg.exp.pred_checkpoint_path is None
        else cfg.exp.pred_checkpoint_path
    )
    model_module = LeapLightningModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    if cfg.exp.ema.use_ema:
        model_module.model = model_module.model_ema.module
    model = model_module.model

    dm = LeapLightningDataModule(cfg)
    dataloader, val2_df = dm.val2_dataloader()
    original_xs = []
    preds = []
    labels = []
    model = model.to("cuda")
    model.eval()
    for x, x_cat, _, _, original_x, original_y in tqdm(dataloader):
        x, x_cat, original_x, original_y = (
            x.to("cuda"),
            x_cat.to("cuda"),
            original_x.to("cuda"),
            original_y.to("cuda"),
        )
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x_cat = torch.flatten(x_cat, start_dim=0, end_dim=1)
        original_x = torch.flatten(original_x, start_dim=0, end_dim=1)
        original_y = torch.flatten(original_y, start_dim=0, end_dim=1)
        with torch.no_grad():
            out = model(x.to(torch_dtype), x_cat.to(torch.long))

        original_xs.append(original_x.cpu().to(torch.float64))
        preds.append(out.cpu().to(torch.float64))
        labels.append(original_y.cpu())
        if cfg.debug:
            break

    with utils.trace("save predict"):
        original_xs = np.concatenate(original_xs, axis=0)
        preds = Scaler(cfg).inv_scale_output(np.concatenate(preds, axis=0), original_xs)
        labels = np.concatenate(labels, axis=0)

        original_xs_df = pd.DataFrame(
            original_xs, columns=[i for i in range(original_xs.shape[1])]
        ).reset_index()
        original_xs_df.to_parquet(output_path / "val2_original_xs.parquet")
        del original_xs_df
        del original_xs
        gc.collect()

        original_predict_df = pd.DataFrame(
            preds, columns=[i for i in range(preds.shape[1])]
        ).reset_index()
        original_predict_df.to_parquet(output_path / "val2_predict.parquet")
        del original_predict_df
        gc.collect()

        original_label_df = pd.DataFrame(
            labels, columns=[i for i in range(labels.shape[1])]
        ).reset_index()
        original_label_df.to_parquet(output_path / "val2_label.parquet")
        del original_label_df
        gc.collect()

    # weight (weight zero もあるのでかけておく)
    ss_df = pl.read_csv(
        "input/leap-atmospheric-physics-ai-climsim/sample_submission.csv", n_rows=1
    )
    weight_array = ss_df.select(
        [x for x in ss_df.columns if x != "sample_id"]
    ).to_numpy()[0]
    predict_df = pd.DataFrame(
        preds * weight_array, columns=[i for i in range(preds.shape[1])]
    ).reset_index()
    label_df = pd.DataFrame(
        labels * weight_array, columns=[i for i in range(labels.shape[1])]
    ).reset_index()
    r2_scores = score(label_df, predict_df, "index", multioutput="raw_values")
    r2_score_dict = {
        col: r2 for col, r2 in dict(zip(cfg.cols.col_names, r2_scores)).items()
    }
    pickle.dump(r2_score_dict, open(output_path / "val2_r2_score_dict.pkl", "wb"))
    r2_score = float(np.array([v for v in r2_score_dict.values()]).mean())
    print(f"{r2_score=}")
    wandb.log(
        {
            "r2_score/val2": r2_score,
        }
    )
    del predict_df, label_df
    gc.collect()

    # save
    val2_df = pl.concat(
        [
            val2_df.select("sample_id"),
            pl.from_numpy(preds * weight_array, schema=ss_df.columns[1:]),
        ],
        how="horizontal",
    )
    print(val2_df)
    val2_df.write_parquet(output_path / "valid_pred.parquet")
    del val2_df


def predict_test(cfg: DictConfig, output_path: Path) -> None:
    torch_dtype = torch.float64 if "64" in cfg.exp.precision else torch.float32
    checkpoint_path = (
        output_path / "checkpoints" / "best_model.ckpt"
        if cfg.exp.pred_checkpoint_path is None
        else cfg.exp.pred_checkpoint_path
    )
    model_module = LeapLightningModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    if cfg.exp.ema.use_ema:
        model_module.model = model_module.model_ema.module
    model = model_module.model

    dm = LeapLightningDataModule(cfg)
    dataloader, test_df = dm.test_dataloader()
    original_xs = []
    preds = []
    model = model.to("cuda")
    model.eval()
    for x, x_cat, original_x in tqdm(dataloader):
        x = x.to("cuda")
        x_cat = x_cat.to("cuda")
        # webdatasetとは違い、batchでの読み出しではないのでflattenは必要ない
        with torch.no_grad():
            out = model(x.to(torch_dtype), x_cat.to(torch.long))
        original_xs.append(original_x.cpu().to(torch.float64).numpy())
        preds.append(out.detach().cpu().numpy())

    original_xs = np.concatenate(original_xs, axis=0)
    preds = Scaler(cfg).inv_scale_output(np.concatenate(preds, axis=0), original_xs)

    # load sample
    sample_submission_df = pl.read_parquet(
        cfg.exp.sample_submission_path,
        n_rows=(None if cfg.debug is False else len(preds)),
    )[: len(preds)]
    preds *= sample_submission_df[:, 1:].to_numpy()

    sample_submission_df = pl.concat(
        [
            sample_submission_df.select("sample_id"),
            pl.from_numpy(preds, schema=sample_submission_df.columns[1:]),
        ],
        how="horizontal",
    )

    sample_submission_df.write_parquet(output_path / "submission.parquet")
    print(sample_submission_df)


def viz(cfg: DictConfig, output_dir: Path, exp_name: str):
    import papermill as pm

    output_notebook_path = str(output_dir / "result_viz.ipynb")
    pm.execute_notebook(
        cfg.exp.viz_notebook_path,
        output_notebook_path,
        parameters={
            "config_dir": "experiments",
            "exp_name": exp_name,
        },
    )
    # htmlに変換してwandbにアップロード
    os.system(
        "jupyter nbconvert --to html --TagRemovePreprocessor.remove_input_tags hide "
        + output_notebook_path
    )
    wandb.log(
        {
            "result_viz": wandb.Html(
                open(output_notebook_path.replace(".ipynb", ".html"))
            )
        }
    )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.exp_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    seed_everything(cfg.exp.seed)

    pl_logger = WandbLogger(
        name=exp_name,
        project="kaggle-leap2",
        mode="disabled" if cfg.debug else None,
    )
    pl_logger.log_hyperparams(cfg)

    if "train" in cfg.exp.modes:
        train(cfg, output_path, pl_logger)
    # if "valid" in cfg.exp.modes:
    #    predict_valid(cfg, output_path)
    if "valid2" in cfg.exp.modes:
        predict_val2(cfg, output_path)
    if "test" in cfg.exp.modes:
        predict_test(cfg, output_path)
    if "viz" in cfg.exp.modes:
        viz(cfg, output_path, exp_name)


if __name__ == "__main__":
    main()
