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
from sklearn.metrics import r2_score
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import wandb

sys.path.append(".")

import utils
from utils.humidity import cal_specific2relative_coef
from utils.metric import score


def get_valid_name(cfg):
    return f"{cfg.exp.valid_start[0]:02d}-{cfg.exp.valid_start[1]:02d}_{cfg.exp.valid_end[0]:02d}-{cfg.exp.valid_end[1]:02d}_{cfg.exp.valid_data_skip_mod}"


def create_board(base=0):
    board = np.zeros((8, 8), dtype=int)
    num = 0
    for i in range(0, 8, 2):
        for j in range(0, 8, 2):
            board[i, j] = num
            board[i, j + 1] = num + 1
            board[i + 1, j] = num + 2
            board[i + 1, j + 1] = num + 3
            num += 4
    return np.fliplr(np.rot90(board, k=2)) + (base * 64)


def get_neighbors(face, i, j):
    neighbors = []
    if i > 0:
        neighbors.append(face[i - 1, j])
    if i < 7:
        neighbors.append(face[i + 1, j])
    if j > 0:
        neighbors.append(face[i, j - 1])
    if j < 7:
        neighbors.append(face[i, j + 1])
    return neighbors


def get_boundary_neighbors(faces):
    boundary_neighbors = {i: [] for i in range(384)}

    # 境界の接続を定義（面1、エッジ1、面2、エッジ2、順序）
    # left ↓、up→, right ↓, down →
    connections = [
        (0, "left", 3, "right", "same"),
        (0, "up", 5, "down", "same"),
        (0, "right", 1, "left", "same"),
        (0, "down", 4, "up", "same"),
        (1, "left", 0, "right", "same"),
        (1, "up", 5, "right", "reverse"),
        (1, "right", 2, "left", "same"),
        (1, "down", 4, "right", "same"),
        (2, "left", 1, "right", "same"),
        (2, "up", 5, "up", "reverse"),
        (2, "right", 3, "left", "same"),
        (2, "down", 4, "down", "reverse"),
        (3, "left", 2, "right", "same"),
        (3, "up", 5, "left", "same"),
        (3, "right", 0, "left", "same"),
        (3, "down", 4, "left", "reverse"),
        (4, "left", 3, "down", "reverse"),
        (4, "up", 0, "down", "same"),
        (4, "right", 1, "down", "same"),
        (4, "down", 2, "down", "reverse"),
        (5, "left", 3, "up", "same"),
        (5, "up", 2, "up", "reverse"),
        (5, "right", 1, "up", "reverse"),
        (5, "down", 0, "up", "same"),
    ]

    def get_edge_coord(index, edge, reverse):
        if edge == "left":
            coord = (index, 0)
        elif edge == "right":
            coord = (index, 7)
        elif edge == "up":
            coord = (0, index)
        elif edge == "down":
            coord = (7, index)
        if reverse:
            if edge in ["left", "right"]:
                coord = (7 - index, coord[1])
            elif edge in ["up", "down"]:
                coord = (coord[0], 7 - index)
        return coord

    for face1, edge1, face2, edge2, order in connections:
        for i in range(8):
            coord1 = get_edge_coord(i, edge1, False)
            coord2 = get_edge_coord(i, edge2, order == "reverse")
            boundary_neighbors[faces[face1][coord1]].append(faces[face2][coord2])
            boundary_neighbors[faces[face2][coord2]].append(faces[face1][coord1])

    return boundary_neighbors


def create_adjacency_matrix(self_loop=True, spectral_connection=True):
    # 隣接行列の作成
    N = 384
    adjacency_matrix = np.zeros((N, N), dtype=int)

    # 各面の生成
    faces = [create_board(base=i) for i in range(6)]

    # 各面内の隣接を追加
    for face in faces:
        for i in range(8):
            for j in range(8):
                idx = face[i, j]
                neighbors = get_neighbors(face, i, j)
                for neighbor in neighbors:
                    adjacency_matrix[idx, neighbor] = 1
                    adjacency_matrix[neighbor, idx] = 1

    # 境界の隣接を追加
    boundary_neighbors = get_boundary_neighbors(faces)
    for idx, neighbors in boundary_neighbors.items():
        for neighbor in neighbors:
            adjacency_matrix[idx, neighbor] = 1
            adjacency_matrix[neighbor, idx] = 1

    assert adjacency_matrix[0, 205] == 1
    assert adjacency_matrix[13, 64] == 1
    assert adjacency_matrix[77, 128] == 1
    assert adjacency_matrix[141, 192] == 1
    assert adjacency_matrix[338, 250] == 1
    assert adjacency_matrix[336, 251] == 1
    assert adjacency_matrix[0, 306] == 1
    assert adjacency_matrix[50, 320] == 1
    assert adjacency_matrix[64, 319] == 1
    assert adjacency_matrix[118, 349] == 1

    assert adjacency_matrix.sum() == 384 * 4

    # 4つずつのまとまりについては spectral element 内で結合しているとみなす
    if spectral_connection:
        for i in range(384):
            for j in range(i + 1, 384):
                if i // 4 == j // 4:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1

    for i in range(384):
        if self_loop:
            adjacency_matrix[i][i] = 1
        else:
            adjacency_matrix[i][i] = 0

    return adjacency_matrix


def create_edge_index(self_loop=True, spectral_connection=True):
    adjacency_matrix = create_adjacency_matrix(self_loop, spectral_connection)
    edge_index = np.array(np.nonzero(adjacency_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index


def create_edge_attr(edge_index, is_same_spectral=True) -> torch.tensor:
    """
    output: (len(edge), 4)
    lat,lonの差をsin,cosで表現して入力
    """
    grid_path = "/kaggle/working/misc/grid_info/ClimSim_low-res_grid-info.nc"
    grid_info = xr.open_dataset(grid_path)
    latitude = grid_info["lat"].to_numpy()
    longitude = grid_info["lon"].to_numpy()
    latitude_radian = np.radians(latitude)
    longitude_radian = np.radians(longitude)

    lat_diff = latitude_radian[edge_index[0, :]] - latitude_radian[edge_index[1, :]]
    lon_diff = longitude_radian[edge_index[0, :]] - longitude_radian[edge_index[1, :]]

    lat_diff_sin = torch.tensor(np.sin(lat_diff))
    lat_diff_cos = torch.tensor(np.cos(lat_diff))
    lon_diff_sin = torch.tensor(np.sin(lon_diff))
    lon_diff_cos = torch.tensor(np.cos(lon_diff))
    is_same_spectral = edge_index[0, :] // 4 == edge_index[1, :] // 4
    edge_attr = torch.stack(
        [lat_diff_sin, lat_diff_cos, lon_diff_sin, lon_diff_cos, is_same_spectral],
        dim=1,
    )
    return edge_attr


# physical constatns from (E3SM_ROOT/share/util/shr_const_mod.F90)
grav = 9.80616  # acceleration of gravity ~ m/s^2
cp = 1.00464e3  # specific heat of dry air   ~ J/kg/K
lv = 2.501e6  # latent heat of evaporation ~ J/kg
lf = 3.337e5  # latent heat of fusion      ~ J/kg
ls = lv + lf  # latent heat of sublimation ~ J/kg
rho_air = 101325.0 / (6.02214e26 * 1.38065e-23 / 28.966) / 273.15
rho_h20 = 1.0e3  # de


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


class LeapLightningDataModule(LightningDataModule):
    def __init__(
        self,
        cfg,
        test_df,
    ):
        super().__init__()
        self.scaler = Scaler(cfg)
        self.cfg = cfg
        self.test_df = test_df
        self.rng = random.Random(self.cfg.exp.seed)
        grid_path = "/kaggle/working/misc/grid_info/ClimSim_low-res_grid-info.nc"
        grid_info = xr.open_dataset(grid_path)
        self.hyai = grid_info["hyai"].to_numpy()
        self.hybi = grid_info["hybi"].to_numpy()

    class TestDataset(Dataset):
        def __init__(self, cfg, test_df, scaler, hyai, hybi):
            self.cfg = cfg
            self.scaler = scaler
            self.hyai = hyai
            self.hybi = hybi
            # 提供データは cam_in_SNOWHICE は削除済みなので削除しないが、idを削除する
            self.x = test_df[:, 1:].to_numpy()

            # 長さは384の倍数にする
            mod = self.x.shape[0] % 384
            if mod > 0:
                self.x = self.x[:-mod]

        def __len__(self):
            return self.x.shape[0] // 384

        def __getitem__(self, index):
            original_x = self.x[index * 384 : (index + 1) * 384]
            x, x_cat = self.scaler.scale_input(original_x)
            return (
                torch.from_numpy(x),
                torch.from_numpy(x_cat),
                torch.from_numpy(original_x),
            )

    def test_dataloader(self):
        self.test_dataset = self.TestDataset(
            self.cfg, self.test_df, self.scaler, self.hyai, self.hybi
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.exp.valid_batch_size,
            num_workers=self.cfg.exp.num_workers,
            shuffle=False,
            pin_memory=False,
        )


class MultiConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2
    height and width size will be changed to size-4.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        layers,
        kernel_sizes,
        use_batch_norm=True,
    ):
        super().__init__()
        self.in_conv = []
        self.in_conv.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                padding="same",
                bias=False,
            )
        )
        if use_batch_norm:
            self.in_conv.append(nn.BatchNorm2d(out_channels))
        self.in_conv.append(nn.ReLU(inplace=True))
        self.in_conv = nn.Sequential(*self.in_conv)

        self.convs = []
        for i, layer in enumerate(layers):
            conv = []
            for _ in range(layer):
                conv.append(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=(kernel_sizes[i], 1),
                        padding="same",
                        bias=False,
                    )
                )
                if use_batch_norm:
                    conv.append(nn.BatchNorm2d(out_channels))
                conv.append(nn.ReLU(inplace=True))
            self.convs.append(nn.Sequential(*conv))
        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        x = self.in_conv(x)
        # skip connection
        for conv in self.convs:
            x = x + conv(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, in_channels, out_channels, layers, kernel_sizes, use_batch_norm=True
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d((2, 1)),
            MultiConv(
                in_channels,
                out_channels,
                layers,
                kernel_sizes,
                use_batch_norm=use_batch_norm,
            ),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        layers,
        kernel_sizes,
        bilinear=True,
        use_batch_norm=True,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=(2, 1), stride=2
        )
        self.conv = MultiConv(
            in_channels,
            out_channels,
            layers,
            kernel_sizes,
            use_batch_norm=use_batch_norm,
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
        layers,
        kernel_sizes,
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

        self.inc = MultiConv(
            n_channels,
            n_base_channels,
            layers,
            kernel_sizes,
            use_batch_norm=use_batch_norm,
        )
        self.downs = nn.ModuleList(
            [
                Down(
                    n_base_channels * (2**i),
                    n_base_channels * (2 ** (i + 1)),
                    layers,
                    kernel_sizes,
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
                    layers,
                    kernel_sizes,
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


class Height60Conv(MessagePassing):
    """
    高さを表す次元は結合した状態じゃないとうまく動かない
    viewやflattenを使って整えてからkernel size 1 の1dCNNで高さを保ったまま処理
    """

    def __init__(self, base_channels, edge_channels, kernel_sizes=[1], bias=False):
        super().__init__(aggr="add")
        self.edge_channels = edge_channels
        self.base_channels = base_channels
        self.edge_linear = nn.Linear(edge_channels, base_channels)

        input_channels = 3 * base_channels
        adjacency_conv_list = []
        for kernel_size in kernel_sizes:
            adjacency_conv_list.append(
                nn.Conv1d(
                    input_channels,
                    base_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=bias,
                )
            )
            adjacency_conv_list.append(nn.LayerNorm((base_channels, 60)))
            adjacency_conv_list.append(nn.ReLU())
            input_channels = base_channels

        self.adjacency_conv = nn.Sequential(*adjacency_conv_list)

    def forward(self, x, edge_attr, edge_index):
        """
        x: (384, base_channels*60)
        edge_attr: (len(edge_index), edge_channels)
        edge_index: (2, 384)  # これは固定
        """

        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x

    def message(self, x_i, x_j, edge_attr):
        """
        x_j: (len(edge), base_channels*60)
        edge_attr: (len(edge), edge_channels)
        """
        h_i = x_i.view(-1, self.base_channels, 60)
        h_j = x_j.view(-1, self.base_channels, 60)
        edge_attr = self.edge_linear(edge_attr)  # (len(edge), base_channels)
        h = torch.concat(
            [h_i, h_j - h_i, edge_attr.unsqueeze(-1).repeat(1, 1, 60)], dim=-2
        )
        h = self.adjacency_conv(h)
        h = h.flatten(start_dim=-2, end_dim=-1)
        return h


class GNN(nn.Module):
    def __init__(
        self,
        base_channels,
        n_layers=4,
        activation="relu",
        kernel_sizes=[1],
        bias=False,
        self_loop=True,
        spectral_connection=True,
        is_same_spectral=True,
    ):
        super().__init__()
        self.edge_index = create_edge_index(
            self_loop=self_loop, spectral_connection=spectral_connection
        )
        self.edge_attr = create_edge_attr(self.edge_index, is_same_spectral).float()
        self.base_channels = base_channels
        self.activation = activation

        edge_channels = self.edge_attr.shape[-1]
        self.conv = nn.ModuleList(
            [
                Height60Conv(base_channels, edge_channels, kernel_sizes, bias=bias)
                for _ in range(n_layers)
            ]
        )

    def get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        """
        x: (384, n_base_channels, 60)
        edge_attr: (len(edge), edge_channels)
        edge_index: (2, len(edge))  # これは固定
        """
        assert x.shape[1] == self.base_channels

        edge_index = self.edge_index.to(x.device)
        edge_attr = self.edge_attr.to(x.device)

        # 一旦高さ次元をまとめて処理。conv内で分けて処理される
        x = x.flatten(start_dim=-2, end_dim=-1)
        for conv in self.conv:
            x = conv(x, edge_attr, edge_index)  # Conv1d適用
        x = x.view(
            -1, self.base_channels, 60
        )  # (384, n_base_channels*60) -> (384, n_base_channels, 60)
        return x


class UnetGNNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels,
        layers,
        kernel_sizes,
        depth,
        dropout,
        gnn_n_layers,
        gnn_kernel_sizes,
        gnn_bias,
        self_loop,
        use_batch_norm,
    ):
        super().__init__()
        self.unet = UNet(
            n_channels=in_channels,
            n_classes=base_channels,
            bottleneck_out_nums=8,
            layers=layers,
            kernel_sizes=kernel_sizes,
            depth=depth,
            dropout=dropout,
            n_base_channels=base_channels,
            use_batch_norm=use_batch_norm,
        )

        self.gnn = GNN(
            base_channels=base_channels,
            n_layers=gnn_n_layers,
            self_loop=self_loop,
            spectral_connection=False,
            is_same_spectral=False,
            kernel_sizes=gnn_kernel_sizes,
            bias=gnn_bias,
        )

    def forward(self, x):
        """
        x: (384, in_channels, 60)
        outは base_channels*2 になる
        """
        x = x.unsqueeze(-1)
        x, bottleneck_feat = self.unet(x)
        x = x.squeeze(-1)  # ->(batch*384, n_base_channels, 60)

        x_out = self.gnn(x)
        x = torch.cat([x, x_out], dim=1)

        return x, bottleneck_feat


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
        n_base_channels=32,
        depth=4,
        unet_depth=4,
        unet_dropout=0.2,
        layers=[1, 1],
        kernel_sizes=[5, 3],
        gnn_n_layers=1,
        gnn_kernel_sizes=[1],
        gnn_bias=False,
        self_loop=True,
        spectral_connection=True,
        is_same_spectral=True,
        seq_feats=[],
        scalar_feats=[],
    ):
        super().__init__()

        self.seq_feats = seq_feats
        self.scalar_feats = scalar_feats
        self.n_base_channels = n_base_channels
        self.same_height_hidden_sizes = same_height_hidden_sizes

        num_embeddings = 60
        self.positional_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.constant_encoder = nn.Linear(1, 60)
        self.categorical_embedding = nn.Embedding(240, categorical_embedding_dim)

        previous_size = 9 + 16 + embedding_dim  # 9: sequence, 16: scalar
        previous_size += len(seq_feats)
        previous_size += len(scalar_feats)
        previous_size += 2 * categorical_embedding_dim
        self.same_height_encoder = MLP(
            previous_size, same_height_hidden_sizes, use_layer_norm=use_input_layer_norm
        )

        self.in_gnn = GNN(
            base_channels=same_height_hidden_sizes[-1],
            n_layers=gnn_n_layers,
            spectral_connection=spectral_connection,
            self_loop=self_loop,
            is_same_spectral=is_same_spectral,
            kernel_sizes=gnn_kernel_sizes,
            bias=gnn_bias,
        )

        unet_gnn_layers = []
        in_channels = 3 * same_height_hidden_sizes[-1]
        for _ in range(depth):
            unet_gnn_layers.append(
                UnetGNNLayer(
                    in_channels=in_channels,
                    base_channels=n_base_channels,
                    layers=layers,
                    kernel_sizes=kernel_sizes,
                    depth=unet_depth,
                    dropout=unet_dropout,
                    gnn_n_layers=gnn_n_layers,
                    gnn_kernel_sizes=gnn_kernel_sizes,
                    gnn_bias=gnn_bias,
                    self_loop=self_loop,
                    use_batch_norm=use_batch_norm,
                )
            )
            in_channels = 2 * n_base_channels
        self.unet_gnn_layers = nn.ModuleList(unet_gnn_layers)

        out_base_channels = 2 * n_base_channels
        self.t_head = MLP(
            9 + out_base_channels,
            output_hidden_sizes + [2],
            use_layer_norm=use_output_layer_norm,
        )
        self.q1_head = MLP(
            5 + out_base_channels,
            output_hidden_sizes + [2],
            use_layer_norm=use_output_layer_norm,
        )
        self.cloud_water_head = MLP(
            6 + out_base_channels,
            output_hidden_sizes + [4],
            use_layer_norm=use_output_layer_norm,
        )
        self.wind_head = nn.ModuleList(
            [
                MLP(
                    8 + out_base_channels,
                    output_hidden_sizes + [2],
                    use_layer_norm=use_output_layer_norm,
                )
                for _ in range(2)
            ]
        )

    def forward(self, x, x_cat):
        """
        x: (batch, 384, dim)
        """

        """
        1. 各地域ごとの処理
        """
        # 一旦 0,1 次元目をまとめて (batch*384, dim) にして処理
        x = x.flatten(start_dim=0, end_dim=1)
        x_cat = x_cat.flatten(start_dim=0, end_dim=1)
        relative_humidity_all = x[:, 556 : 556 + 60].unsqueeze(-1)
        x_cloud_snow_rate_array = x[:, 556 : 556 + 60].unsqueeze(-1)
        x_cloud_water = x[:, 556 + 60 : 556 + 120].unsqueeze(-1)
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

        # (batch*384, 120) -> (batch*384, 120, 5)
        x_cat = self.categorical_embedding(x_cat)
        # (batch*384, 120, 5) -> (batch*384, 60, 10)
        x_cat = torch.cat(
            [
                x_cat[:, :60, :],
                x_cat[:, 60:120, :],
            ],
            dim=2,
        )
        input_list.append(x_cat)
        x = torch.cat(
            input_list,
            dim=2,
        )  #  (batch*384, 60, dim)

        # (batch*384, 60, dim) -> (batch*384, 60, same_height_hidden_sizes[-1])
        x = self.same_height_encoder(x)
        x = x.transpose(-1, -2)  # ->(batch*384, same_height_hidden_sizes[-1], 60)

        """
        batch = 1 を仮定しているので注意
        """
        x_out = self.in_gnn(x)
        x = torch.cat([x, x_out], dim=1)

        # global の平均を取って結合
        x_global = x[:, : self.same_height_hidden_sizes[-1], :].mean(dim=0)
        x_global_diff = x[:, : self.same_height_hidden_sizes[-1], :] - x_global
        x = torch.cat([x, x_global_diff], dim=1)

        """
        """
        for unet_gnn_layer in self.unet_gnn_layers:
            x, bottleneck_feat = unet_gnn_layer(x)

        x = x.transpose(-1, -2)  # ->(batch*384, 60, n_base_channels)
        """
        3. 各地域ごとに出力を作成
        """
        out_t = self.t_head(
            torch.cat(
                [
                    x_state_t,
                    relative_humidity_all,
                    x_state_q0001,
                    x_state_q0002,
                    x_state_q0003,
                    x_cloud_water,
                    x_cloud_snow_rate_array,
                    x_state_u,
                    x_state_v,
                    x,
                ],
                dim=2,
            )
        )  # -> (batch, 60, 1)
        out_t = out_t[:, :, 0:1].exp() - out_t[:, :, 1:2].exp()

        out_q1 = self.q1_head(
            torch.cat(
                [
                    x_state_t,
                    out_t,
                    relative_humidity_all,
                    x_state_q0001,
                    x_cloud_water,
                    x,
                ],
                dim=2,
            )
        )
        out_q1 = out_q1[:, :, 0:1].exp() - out_q1[:, :, 1:2].exp()

        out_cw = self.cloud_water_head(
            torch.cat(
                [
                    x_state_t,
                    out_t,
                    relative_humidity_all,
                    x_state_q0001,
                    x_cloud_water,
                    x_cloud_snow_rate_array,
                    x,
                ],
                dim=2,
            )
        )
        out_q2 = out_cw[:, :, 0:1].exp() - out_cw[:, :, 1:2].exp()
        out_q3 = out_cw[:, :, 2:3].exp() - out_cw[:, :, 3:4].exp()

        out_u = self.wind_head[0](
            torch.cat(
                [
                    x_state_t,
                    out_t,
                    x_state_q0001,
                    x_state_q0002,
                    x_state_q0003,
                    x_cloud_water,
                    x_cloud_snow_rate_array,
                    x_state_u,
                    x,
                ],
                dim=2,
            )
        )
        out_u = out_u[:, :, 0:1].exp() - out_u[:, :, 1:2].exp()

        out_v = self.wind_head[1](
            torch.cat(
                [
                    x_state_t,
                    out_t,
                    x_state_q0001,
                    x_state_q0002,
                    x_state_q0003,
                    x_cloud_water,
                    x_cloud_snow_rate_array,
                    x_state_v,
                    x,
                ],
                dim=2,
            )
        )
        out_v = out_v[:, :, 0:1].exp() - out_v[:, :, 1:2].exp()

        out = torch.cat([out_t, out_q1, out_q2, out_q3, out_u, out_v], dim=2)
        out = out.transpose(-1, -2)
        out = out.flatten(start_dim=1)
        out = torch.cat([out, bottleneck_feat], dim=1)

        # 最後に戻す
        out = out.view(-1, 384, 368)
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
        # self.loss_fc = nn.MSELoss()  # Using MSE for regression
        # self.loss_fc = nn.L1Loss()
        self.loss_fc = nn.SmoothL1Loss(beta=cfg.exp.l1_beta)
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

        ss_df = pl.read_parquet(cfg.exp.sample_submission_path, n_rows=1)
        self.weight_array = ss_df.select(
            [x for x in ss_df.columns if x != "sample_id"]
        ).to_numpy()[0]

    def training_step(self, batch, batch_idx):
        mode = "train"
        x, x_cat, y, mask, _, _ = batch
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
        del x, x_cat, y, out, out_masked, y_masked
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        mode = "valid"
        x, x_cat, y, mask, original_x, original_y = batch
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

        # バッチごとに分割されているので結合
        out = np.concatenate(out.detach().cpu().numpy(), axis=0)
        original_y = np.concatenate(original_y.cpu().to(torch.float64).numpy(), axis=0)
        original_x = np.concatenate(original_x.cpu().to(torch.float64).numpy(), axis=0)
        self.valid_preds.append(out)
        self.valid_labels.append(original_y)
        self.valid_original_xs.append(original_x)
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


def predict_test(cfg: DictConfig, output_path: Path) -> None:
    torch_dtype = torch.float64 if "64" in cfg.exp.precision else torch.float32
    checkpoint_path = (
        output_path / "checkpoints" / "best_model.ckpt"
        if cfg.exp.pred_checkpoint_path is None
        else cfg.exp.pred_checkpoint_path
    )
    model_module = LeapLightningModule.load_from_checkpoint(
        checkpoint_path, map_location="cuda", cfg=cfg
    )
    if cfg.exp.ema.use_ema:
        model_module.model = model_module.model_ema.module
    model = model_module.model

    test_info_df = pl.read_parquet(cfg.exp.test_info_path)
    original2sort_index = test_info_df.sort("sort_index")["original_index"].to_numpy()

    test_df = pl.read_parquet(
        cfg.exp.test_path, n_rows=(None if cfg.debug is False else 500)
    )
    test_df = test_df[original2sort_index]
    print(test_df)

    dm = LeapLightningDataModule(cfg, test_df)
    dataloader = dm.test_dataloader()
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
        out = np.concatenate(out.detach().cpu().numpy(), axis=0)
        original_x = np.concatenate(original_x.cpu().to(torch.float64).numpy(), axis=0)
        original_xs.append(original_x)
        preds.append(out)

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

    # 末尾の足りない部分を結合する
    fill_submission_df = pl.read_parquet(
        cfg.exp.fill_submission_path,
        n_rows=(None if cfg.debug is False else len(test_df)),
    )[len(preds) :]
    sample_submission_df = pl.concat(
        [
            sample_submission_df,
            fill_submission_df,
        ],
        how="vertical",
    )

    # 元の順番に戻す
    sort2original_index = test_info_df.sort("original_index")["sort_index"].to_numpy()
    sample_submission_df = sample_submission_df[sort2original_index]

    sample_submission_df = sample_submission_df.with_columns(test_info_df["sample_id"])

    sample_submission_df.write_parquet(output_path / "submission.parquet")
    print(sample_submission_df)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.exp_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    seed_everything(cfg.exp.seed)

    if "test" in cfg.exp.modes:
        predict_test(cfg, output_path)


if __name__ == "__main__":
    main()
