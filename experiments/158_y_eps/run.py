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

import utils
import wandb
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


def get_neighbors(face, i, j, diag_edge):
    neighbors = []
    if i > 0:
        neighbors.append(face[i - 1, j])
    if i < 7:
        neighbors.append(face[i + 1, j])
    if j > 0:
        neighbors.append(face[i, j - 1])
    if j < 7:
        neighbors.append(face[i, j + 1])
    # 斜め
    if diag_edge:
        if i > 0 and j > 0:
            neighbors.append(face[i - 1, j - 1])
        if i > 0 and j < 7:
            neighbors.append(face[i - 1, j + 1])
        if i < 7 and j > 0:
            neighbors.append(face[i + 1, j - 1])
        if i < 7 and j < 7:
            neighbors.append(face[i + 1, j + 1])

    return neighbors


def get_boundary_neighbors(faces, diag_edge):
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
            if diag_edge:
                if i >= 1:
                    coord2 = get_edge_coord(i - 1, edge2, order == "reverse")
                    boundary_neighbors[faces[face1][coord1]].append(
                        faces[face2][coord2]
                    )
                    boundary_neighbors[faces[face2][coord2]].append(
                        faces[face1][coord1]
                    )
                if i <= 6:
                    coord2 = get_edge_coord(i + 1, edge2, order == "reverse")
                    boundary_neighbors[faces[face1][coord1]].append(
                        faces[face2][coord2]
                    )
                    boundary_neighbors[faces[face2][coord2]].append(
                        faces[face1][coord1]
                    )

    return boundary_neighbors


def create_adjacency_matrix(
    self_loop=True, use_boundary=True, spectral_connection=True, diag_edge=False
):
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
                neighbors = get_neighbors(face, i, j, diag_edge=diag_edge)
                for neighbor in neighbors:
                    adjacency_matrix[idx, neighbor] = 1
                    adjacency_matrix[neighbor, idx] = 1

    # 境界の隣接を追加
    if use_boundary:
        boundary_neighbors = get_boundary_neighbors(faces, diag_edge)
        for idx, neighbors in boundary_neighbors.items():
            for neighbor in neighbors:
                adjacency_matrix[idx, neighbor] = 1
                adjacency_matrix[neighbor, idx] = 1
    """
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
    """

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


def create_edge_index(
    self_loop=True, use_boundary=True, spectral_connection=True, diag_edge=False
):
    adjacency_matrix = create_adjacency_matrix(
        self_loop, use_boundary, spectral_connection, diag_edge=diag_edge
    )
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
            self.cfg.exp.y_eps,
        )

        grid_path = "/kaggle/working/misc/grid_info/ClimSim_low-res_grid-info.nc"
        grid_info = xr.open_dataset(grid_path)
        self.hyai = grid_info["hyai"].to_numpy()
        self.hybi = grid_info["hybi"].to_numpy()
        self.hyam = grid_info["hyam"].to_numpy()
        self.hybm = grid_info["hybm"].to_numpy()

        self.fill_target_index = [
            cfg.cols.col_names.index(col) for col in cfg.exp.fill_target
        ]

        self.y_max_min = np.load(
            Path(cfg.exp.bound_dir) / "y_sample_max_min.npy", allow_pickle=True
        )[557:]
        self.y_min_max = np.load(
            Path(cfg.exp.bound_dir) / "y_sample_min_max.npy", allow_pickle=True
        )[557:]

        self.y_diff = self.y_max_min - self.y_min_max

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
        if "water_energy" in self.cfg.exp.seq_feats:
            water_energy_array = water_array * pressures_array * lv / grav
            feats.append(water_energy_array)
        if "temp_energy" in self.cfg.exp.seq_feats:
            temp_energy_array = x_array[:, 0:60] * pressures_array * cp / grav
            feats.append(temp_energy_array)
        if "temp_water_energy" in self.cfg.exp.seq_feats:
            temp_water_energy_array = water_energy_array + temp_energy_array
            feats.append(temp_water_energy_array)
        if "u_energy" in self.cfg.exp.seq_feats:
            u_energy_array = x_array[:, 240:300] * pressures_array / grav
            feats.append(u_energy_array)
        if "v_energy" in self.cfg.exp.seq_feats:
            v_energy_array = x_array[:, 300:360] * pressures_array / grav
            feats.append(v_energy_array)
        if "wind_energy" in self.cfg.exp.seq_feats:
            wind_energy_array = x_array[:, 240:300] ** 2 + x_array[:, 300:360] ** 2
            feats.append(wind_energy_array)
        if "wind_vec" in self.cfg.exp.seq_feats:
            wind_vec_array = np.sqrt(wind_energy_array)
            feats.append(wind_vec_array)

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
            x[:, start:end] = (
                x[:, start:end] - self.feat_mean_dict[key]
            ) / self.feat_std_dict[key]
        for i, key in enumerate(self.cfg.exp.scalar_feats):
            start = 556 + len(self.cfg.exp.seq_feats) * 60 + i
            end = 556 + len(self.cfg.exp.seq_feats) * 60 + i + 1
            x[:, start:end] = (
                x[:, start:end] - self.feat_mean_dict[key]
            ) / self.feat_std_dict[key]

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
        # fill target はすべて置き換える
        y[:, self.fill_target_index] = original_x[:, self.fill_target_index] * (
            -1 / 1200
        )
        return y

    def filter_and_scale(self, x, y):
        if self.cfg.exp.bound_diff_rate is not None:
            diff = self.y_diff * self.cfg.exp.bound_diff_rate
            filter_bool = np.all(self.y_min_max - diff <= y, axis=1) & np.all(
                y <= self.y_max_min + diff, axis=1
            )
        else:
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
    ):
        super().__init__()
        self.scaler = Scaler(cfg)
        self.cfg = cfg
        self.rng = random.Random(self.cfg.exp.seed)
        self.train_dataset = self._make_dataset("train")
        self.valid_dataset = self._make_dataset("valid")
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
            x, x_cat = self.scaler.scale_input(original_x)
            return (
                torch.from_numpy(x),
                torch.from_numpy(x_cat),
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
        return wds.WebLoader(
            self.valid_dataset,
            batch_size=None,
            num_workers=self.cfg.exp.num_workers,
        ).batched(
            batchsize=self.cfg.exp.valid_batch_size,
            partial=False,
        )

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
            batch_size=self.cfg.exp.valid_batch_size,
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
                x, x_cat, y, filter_bool = self.scaler.filter_and_scale(
                    original_x, original_y
                )
                x, x_cat, y, filter_bool, original_x, original_y = (
                    torch.tensor(x),
                    torch.tensor(x_cat),
                    torch.tensor(y),
                    torch.tensor(filter_bool),
                    torch.tensor(original_x),
                    torch.tensor(original_y),
                )
                yield x, x_cat, y, filter_bool, original_x, original_y

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
        use_boundary=True,
        diag_edge=False,
        spectral_connection=True,
        is_same_spectral=True,
    ):
        super().__init__()
        self.edge_index = create_edge_index(
            self_loop=self_loop,
            use_boundary=use_boundary,
            spectral_connection=spectral_connection,
            diag_edge=diag_edge,
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
        dropout=0.2,
        n_base_channels=32,
        gnn_n_layers1=1,
        gnn_n_layers2=1,
        gnn_kernel_sizes1=[1],
        gnn_kernel_sizes2=[1],
        gnn_bias=False,
        self_loop=True,
        use_boundary=True,
        diag_edge=False,
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

        self.gnn1 = GNN(
            base_channels=same_height_hidden_sizes[-1],
            n_layers=gnn_n_layers1,
            use_boundary=use_boundary,
            diag_edge=diag_edge,
            spectral_connection=spectral_connection,
            self_loop=self_loop,
            is_same_spectral=is_same_spectral,
            kernel_sizes=gnn_kernel_sizes1,
            bias=gnn_bias,
        )
        self.unet = UNet(
            n_channels=3 * same_height_hidden_sizes[-1],
            n_classes=n_base_channels,
            bottleneck_out_nums=8,
            depth=depth,
            dropout=dropout,
            n_base_channels=n_base_channels,
            use_batch_norm=use_batch_norm,
        )

        self.gnn2 = GNN(
            base_channels=n_base_channels,
            n_layers=gnn_n_layers2,
            use_boundary=use_boundary,
            diag_edge=diag_edge,
            self_loop=self_loop,
            spectral_connection=spectral_connection,
            is_same_spectral=is_same_spectral,
            kernel_sizes=gnn_kernel_sizes2,
            bias=gnn_bias,
        )

        out_base_channels = 2 * n_base_channels
        self.t_head = MLP(
            9 + out_base_channels,
            output_hidden_sizes + [1],
            use_layer_norm=use_output_layer_norm,
        )
        self.q1_head = MLP(
            5 + out_base_channels,
            output_hidden_sizes + [1],
            use_layer_norm=use_output_layer_norm,
        )
        self.cloud_water_head = MLP(
            6 + out_base_channels,
            output_hidden_sizes + [2],
            use_layer_norm=use_output_layer_norm,
        )
        self.wind_head = nn.ModuleList(
            [
                MLP(
                    8 + out_base_channels,
                    output_hidden_sizes + [1],
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
        x_out = self.gnn1(x)
        x = torch.cat([x, x_out], dim=1)

        # global の平均を取って結合
        x_global = x[:, : self.same_height_hidden_sizes[-1], :].mean(dim=0)
        x_global_diff = x[:, : self.same_height_hidden_sizes[-1], :] - x_global
        x = torch.cat([x, x_global_diff], dim=1)

        """
        2. unet
        """
        x = x.unsqueeze(-1)
        x, class_logits = self.unet(x)
        x = x.squeeze(-1)  # ->(batch*384, n_base_channels, 60)

        """
        batch = 1 を仮定しているので注意
        """
        x_out = self.gnn2(x)
        x = torch.cat([x, x_out], dim=1)

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

        out = torch.cat([out_t, out_q1, out_cw, out_u, out_v], dim=2)
        out = out.transpose(-1, -2)
        out = out.flatten(start_dim=1)
        out = torch.cat([out, class_logits], dim=1)

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
        self.loss_fc = None
        if cfg.exp.loss.name == "L1Loss":
            self.loss_fc = nn.L1Loss()
        elif cfg.exp.loss.name == "SmoothL1Loss":
            self.loss_fc = nn.SmoothL1Loss()
        elif cfg.exp.loss.name == "MSELoss":
            self.loss_fc = nn.MSELoss()
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
        x, x_cat, y, filter_bool, _, _ = batch
        x, y = (
            x.to(self.torch_dtype),
            y.to(self.torch_dtype),
        )
        x_cat = x_cat.to(torch.long)
        out = self.__pred(x, x_cat, mode)

        out_filtered = out[filter_bool]
        y_filtered = y[filter_bool]
        loss = self.loss_fc(
            out_filtered[:, self.use_cols_index], y_filtered[:, self.use_cols_index]
        )
        self.log(
            f"{mode}_loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        del x, x_cat, y, out, out_filtered, y_filtered
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        mode = "valid"
        x, x_cat, y, _, original_x, original_y = batch
        x, y = (
            x.to(self.torch_dtype),
            y.to(self.torch_dtype),
        )
        x_cat = x_cat.to(torch.long)
        out = self.__pred(x, x_cat, mode)
        loss = self.loss_fc(out[:, self.use_cols_index], y[:, self.use_cols_index])
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

    # Learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # manually warm up lr without a scheduler
        if (
            self.cfg.exp.scheduler.name == "ReduceLROnPlateau"
            and self.trainer.global_step < 5000
        ):
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 5000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.exp.optimizer.lr

        # update params
        optimizer.step(closure=optimizer_closure)

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
    if cfg.exp.restart_ckpt_path:
        model = LeapLightningModule.load_from_checkpoint(
            cfg.exp.restart_ckpt_path, cfg=cfg
        )
    else:
        model = LeapLightningModule(cfg)
    checkpoint_cb = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        verbose=True,
        monitor=monitor,
        mode="max",
        save_top_k=3,
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
    for x, x_cat, original_x, original_y in tqdm(dataloader):
        x, x_cat = (
            x.to("cuda"),
            x_cat.to("cuda"),
        )
        with torch.no_grad():
            out = model(x.to(torch_dtype), x_cat.to(torch.long))

        # バッチごとに分割されているので結合
        out = np.concatenate(out.detach().cpu().numpy(), axis=0)
        original_x = np.concatenate(original_x.cpu().to(torch.float64).numpy(), axis=0)
        original_y = np.concatenate(original_y.cpu().to(torch.float64).numpy(), axis=0)
        original_xs.append(original_x)
        preds.append(out)
        labels.append(original_y)

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
    if "valid" in cfg.exp.modes:
        predict_valid(cfg, output_path)
    if "valid2" in cfg.exp.modes:
        predict_val2(cfg, output_path)
    if "test" in cfg.exp.modes:
        predict_test(cfg, output_path)
    if "viz" in cfg.exp.modes:
        viz(cfg, output_path, exp_name)


if __name__ == "__main__":
    main()
