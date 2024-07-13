import gc
import glob
import itertools
import json
import math
import os
import pickle
import random
import re
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
import xarray as xr
from adan import Adan
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
from sklearn.metrics import (
    mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import utils
import wandb
from utils.humidity import cal_specific2relative_coef


def get_valid_name(n_fold, now_fold):
    return f"{n_fold}fold_{now_fold}fold"


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


class ElapsDataset(Dataset):
    def __init__(self, cfg, scaler, X, y):
        self.cfg = cfg
        self.scaler = scaler
        self.X = X
        self.y = y

    def __len__(self):
        return (len(self.X) + 383) // 384

    def __getitem__(self, index):
        start_index = index * 384
        end_index = min((index + 1) * 384, len(self.X))
        original_x = self.X[start_index:end_index]
        x, x_cat = self.scaler.scale_input(original_x)
        y = self.y[start_index:end_index]

        return (
            torch.from_numpy(x),
            torch.from_numpy(x_cat),
            y,
        )


class TestElapsDataset(Dataset):
    def __init__(self, cfg, scaler, X):
        self.cfg = cfg
        self.scaler = scaler
        self.X = X

    def __len__(self):
        return (len(self.X) + 383) // 384

    def __getitem__(self, index):
        start_index = index * 384
        end_index = min((index + 1) * 384, len(self.X))
        original_x = self.X[start_index:end_index]
        x, x_cat = self.scaler.scale_input(original_x)

        return (
            torch.from_numpy(x),
            torch.from_numpy(x_cat),
        )


class LeapLightningDataModule(LightningDataModule):
    def __init__(
        self,
        cfg,
        X_train,
        X_valid,
        y_train,
        y_valid,
    ):
        super().__init__()
        self.scaler = Scaler(cfg)
        self.cfg = cfg
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def train_dataloader(self):
        train_dataset = ElapsDataset(
            self.cfg,
            self.scaler,
            self.X_train,
            self.y_train,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.cfg.exp.train_batch_size,
            num_workers=self.cfg.exp.num_workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self):
        valid_dataset = ElapsDataset(
            self.cfg,
            self.scaler,
            self.X_valid,
            self.y_valid,
        )
        return DataLoader(
            valid_dataset,
            batch_size=self.cfg.exp.valid_batch_size,
            num_workers=self.cfg.exp.num_workers,
            shuffle=False,
            pin_memory=False,
        )


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
        use_input_layer_norm=False,
        use_output_layer_norm=True,
        embedding_dim=5,
        categorical_embedding_dim=5,
        layers=2,
        seq_feats=[],
        scalar_feats=[],
    ):
        super().__init__()
        self.seq_feats = seq_feats
        self.scalar_feats = scalar_feats
        self.positional_embedding = nn.Embedding(60, embedding_dim)
        self.positional_embedding_last = nn.Embedding(60, embedding_dim)
        self.constant_encoder = nn.Linear(1, 60)
        self.categorical_embedding = nn.Embedding(240, categorical_embedding_dim)

        previous_size = 9 + 16 + embedding_dim
        previous_size += len(seq_feats)
        previous_size += len(scalar_feats)
        previous_size += 2 * categorical_embedding_dim

        self.same_height_encoder = MLP(
            previous_size,
            same_height_hidden_sizes,
            use_layer_norm=use_input_layer_norm,
        )

        input_dim = same_height_hidden_sizes[-1] * 60
        self.head = MLP(
            input_dim,
            [input_dim // (2**i) for i in range(layers)] + [1],
            use_layer_norm=use_output_layer_norm,
        )

    def _preprocess(self, x, x_cat):
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
        return x

    def forward(self, x, x_cat):
        x = self._preprocess(x, x_cat)
        # (batch, 60, dim) -> (batch, 60, same_height_hidden_sizes[-1]*n_feat_channels)
        x = self.same_height_encoder(x)
        x = x.flatten(1, 2)

        out = self.head(x).squeeze(-1).sigmoid()
        return out


class LeapLightningModule(LightningModule):
    def __init__(self, cfg, fold_id):
        super().__init__()
        self.cfg = cfg
        self.model = LeapModel(
            **cfg.exp.model,
            seq_feats=cfg.exp.seq_feats,
            scalar_feats=cfg.exp.scalar_feats,
        )
        self.scaler = Scaler(cfg)
        self.loss_fc = nn.MSELoss()  # Using MSE for regression
        # self.loss_fc = nn.L1Loss()
        # self.bceloss = nn.BCEWithLogitsLoss()
        # self.loss_fc = nn.CrossEntropyLoss()

        self.valid_name = get_valid_name(cfg.exp.n_fold, fold_id)
        self.torch_dtype = torch.float64 if "64" in cfg.exp.precision else torch.float32

        self.best_score = np.inf
        self.best_valid_preds = []
        self.valid_preds = []
        self.valid_y = []

        ss_df = pl.read_parquet(cfg.exp.sample_submission_path, n_rows=1)
        self.weight_array = ss_df.select(
            [x for x in ss_df.columns if x != "sample_id"]
        ).to_numpy()[0]

    def training_step(self, batch, batch_idx):
        mode = "train"
        x, x_cat, y = batch
        x, x_cat, y = (
            torch.flatten(x, start_dim=0, end_dim=1),
            torch.flatten(x_cat, start_dim=0, end_dim=1),
            torch.flatten(y, start_dim=0, end_dim=1),
        )
        x, y = (
            x.to(self.torch_dtype),
            y.to(self.torch_dtype),
        )
        x_cat = x_cat.to(torch.long)
        out = self.__pred(x, x_cat, mode)
        loss = self.loss_fc(out, y)

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
        x, x_cat, y = batch
        x, x_cat, y = (
            torch.flatten(x, start_dim=0, end_dim=1),
            torch.flatten(x_cat, start_dim=0, end_dim=1),
            torch.flatten(y, start_dim=0, end_dim=1),
        )
        x, y = (
            x.to(self.torch_dtype),
            y.to(self.torch_dtype),
        )
        x_cat = x_cat.to(torch.long)
        out = self.__pred(x, x_cat, mode)
        loss = self.loss_fc(out, y)

        self.log(
            f"{mode}_loss/{self.valid_name}",
            loss.detach().item(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        self.valid_preds.append(out.detach().cpu().to(torch.float64).numpy())
        self.valid_y.append(y.detach().cpu().to(torch.float64).numpy())
        return loss

    def predict_step(self, batch, batch_idx):
        mode = "test"
        x, x_cat, _ = batch
        x = x.to(self.torch_dtype)
        x_cat = x_cat.to(torch.long)
        out = self.__pred(x, x_cat, mode)
        return out

    def __pred(self, x, x_cat, mode: str) -> torch.Tensor:
        out = self.model(x, x_cat)
        return out

    def on_validation_epoch_end(self):
        valid_preds = np.concatenate(self.valid_preds, axis=0).astype(np.float64)
        valid_y = np.concatenate(self.valid_y, axis=0).astype(np.float64)
        # scoring mse
        mse_score = mean_squared_error(valid_y, valid_preds)
        self.log(
            f"valid_mse/{self.valid_name}",
            mse_score,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        if mse_score < self.best_score:
            self.best_score = mse_score
            self.best_valid_preds = valid_preds.copy()

        self.valid_preds = []
        self.valid_y = []
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


def train(cfg: DictConfig, output_path: Path, pl_logger) -> None:
    test_old_df = pl.read_parquet(
        cfg.exp.test_old_path, n_rows=None if cfg.debug is False else 384 * 5
    )
    X = test_old_df[:, 1:557].to_numpy()
    y = (
        test_old_df[:, 0:1]
        .with_row_index("elaps")
        .with_columns(pl.col("elaps") / len(test_old_df))["elaps"]
        .to_numpy()
    )
    oof = np.zeros_like(y)
    time_group = test_old_df.with_row_index()["index"].to_numpy() // 384

    # cross validation
    folds = GroupKFold(n_splits=cfg.exp.n_fold)
    for fold_id, (train_index, valid_index) in enumerate(
        folds.split(X, groups=time_group)
    ):
        print(f"fold_id: {fold_id}", len(train_index), len(valid_index))
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        monitor = f"valid_mse/{get_valid_name(cfg.exp.n_fold, fold_id)}"
        dm = LeapLightningDataModule(cfg, X_train, X_valid, y_train, y_valid)
        model = LeapLightningModule(cfg, fold_id)
        checkpoint_cb = ModelCheckpoint(
            dirpath=output_path / f"checkpoints_{fold_id}",
            verbose=True,
            monitor=monitor,
            mode="min",
            save_top_k=2,
            save_last=False,
            enable_version_counter=False,
        )
        lr_monitor = LearningRateMonitor("epoch")
        progress_bar = RichProgressBar()  # leave=True
        model_summary = RichModelSummary(max_depth=2)
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=cfg.exp.early_stopping_patience,
            mode="min",
        )
        if cfg.debug:
            cfg.exp.max_epochs = 2
        trainer = Trainer(
            default_root_dir=output_path,
            accelerator=cfg.exp.accelerator,
            precision=cfg.exp.precision,
            max_epochs=cfg.exp.max_epochs,
            gradient_clip_val=cfg.exp.gradient_clip_val,
            accumulate_grad_batches=cfg.exp.accumulate_grad_batches,
            logger=pl_logger,
            log_every_n_steps=1,
            limit_train_batches=None if cfg.debug is False else 4,
            limit_val_batches=None if cfg.debug is False else 4,
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

        oof[valid_index] = model.best_valid_preds

        shutil.copy(
            checkpoint_cb.best_model_path,
            output_path / f"checkpoints_{fold_id}" / "best_model.ckpt",
        )
        del model, dm, trainer
        gc.collect()
        torch.cuda.empty_cache()

    # save oof
    oof_df = pd.DataFrame(oof, columns=["pred"])
    oof_df.to_parquet(output_path / "oof.parquet")


def predict_test(cfg: DictConfig, output_path: Path) -> None:
    torch_dtype = torch.float64 if "64" in cfg.exp.precision else torch.float32

    test_new_df = pl.read_parquet(
        cfg.exp.test_new_path, n_rows=None if cfg.debug is False else 500
    )
    X = test_new_df[:, 1:557].to_numpy()
    preds_list = []

    for fold_id in range(cfg.exp.n_fold):
        checkpoint_path = (
            output_path / f"checkpoints_{fold_id}" / "best_model.ckpt"
            if cfg.exp.pred_checkpoint_path is None
            else cfg.exp.pred_checkpoint_path
        )
        model_module = LeapLightningModule.load_from_checkpoint(
            checkpoint_path, cfg=cfg, fold_id=fold_id
        )
        dataset = TestElapsDataset(cfg, model_module.scaler, X)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.exp.valid_batch_size,
            num_workers=cfg.exp.num_workers,
            shuffle=False,
            pin_memory=False,
        )

        preds = []
        model = model_module.model
        model = model.to("cuda")
        model.eval()
        for x, x_cat in tqdm(dataloader):
            x, x_cat = (
                torch.flatten(x, start_dim=0, end_dim=1),
                torch.flatten(x_cat, start_dim=0, end_dim=1),
            )
            x = x.to("cuda").to(torch_dtype)
            x_cat = x_cat.to("cuda").to(torch.long)
            with torch.no_grad():
                out = model(x, x_cat)
            preds.append(out.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        preds_list.append(preds)

    preds = np.mean(preds_list, axis=0)
    original_predict_df = pd.DataFrame(preds, columns=["pred"])
    original_predict_df.to_parquet(output_path / "test_predict.parquet")
    print(original_predict_df)


def save_year(cfg, output_path):
    # base
    test_info_df = pl.read_parquet(
        "output/preprocess/make_hack_map/base/test_info.parquet"
    )
    # pred
    test_pred_df = pl.read_parquet(
        "output/experiments/810_pred_elaps_time/base/test_predict.parquet"
    )

    # preprocess
    test_info_df = test_info_df.with_columns(
        [
            (
                pl.datetime(2000, pl.col("month"), pl.col("day"), 0, 0, 0)
                + pl.duration(seconds=pl.col("seconds"))
            ).alias("no_year_timestamp"),
            test_pred_df["pred"],
        ]
    )
    test_info_year_df = (
        (
            test_info_df.with_columns(
                [
                    (
                        pl.count("month").over(["no_year_timestamp", "location"]) == 2
                    ).alias("is_duplicated"),
                    (pl.mean("pred").over(["no_year_timestamp", "location"])).alias(
                        "pred_mean"
                    ),
                    (
                        pl.max("pred").over(["no_year_timestamp", "location"])
                        - pl.min("pred").over(["no_year_timestamp", "location"])
                    ).alias("pred_diff"),
                ]
            )
            .with_columns(
                pl.when(pl.col("is_duplicated") == False)
                .then(pl.lit(0))
                .when(pl.col("pred") > pl.col("pred_mean"))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("is_second_year")
            )
            .with_columns(
                pl.when((pl.col("is_second_year") == 0) & (pl.col("is_duplicated")))
                .then(pl.lit(9))
                .when(
                    (pl.col("is_second_year") == 0)
                    & (~pl.col("is_duplicated"))
                    & (pl.col("month") >= 3)
                )
                .then(pl.lit(9))
                .when(
                    (pl.col("is_second_year") == 0)
                    & (~pl.col("is_duplicated"))
                    & (pl.col("month") < 3)
                )
                .then(pl.lit(10))
                .when(
                    (pl.col("is_second_year") == 1)
                    & (pl.col("is_duplicated"))
                    & (pl.col("month") >= 3)
                )
                .then(pl.lit(10))
                .otherwise(pl.lit(11))
                .alias("year")
            )
            .with_columns(
                (
                    pl.datetime(
                        pl.col("year").cast(pl.Int64),
                        pl.col("month"),
                        pl.col("day"),
                        0,
                        0,
                        0,
                    )
                    + pl.duration(seconds=pl.col("seconds"))
                ).alias("timestamp"),
            )
            .drop(["pred_mean", "no_year_timestamp"])
        )
        .with_row_index("original_index")
        .sort(["timestamp", "location"])
        .with_row_index("sort_index")
    ).sort("original_index")

    test_info_year_df.write_parquet(output_path / "test_info_year.parquet")
    print(test_info_year_df)


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
        project="kaggle-leap-time-binary",
        mode="disabled" if cfg.debug else None,
    )
    pl_logger.log_hyperparams(cfg)

    if "train" in cfg.exp.modes:
        train(cfg, output_path, pl_logger)
    if "test" in cfg.exp.modes:
        predict_test(cfg, output_path)
    if "save" in cfg.exp.modes:
        save_year(cfg, output_path)


if __name__ == "__main__":
    main()
