import json
import os
import pickle
import shutil
import sys
import time
from glob import glob
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import polars as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

TARGET_COLUMNS = pl.read_csv(
    "input/leap-atmospheric-physics-ai-climsim/sample_submission.csv", n_rows=1
).columns[1:]


def base_split(cfg):
    valid_output_dir = Path(cfg.exp.output_dir) / "label" / "valid"
    valid_output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(cfg.exp.valid_path)
    for i in range(0, cfg.exp.n_data_for_eval, 384):
        sub_array = df[
            -cfg.exp.n_data_for_eval + i * 384 : -cfg.exp.n_data_for_eval
            + (i + 1) * 384
        ].to_numpy()
        np.save(valid_output_dir / f"{i:06d}.npy", sub_array)

    output_dir = Path(cfg.exp.output_dir) / "label" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(cfg.exp.test_path)
    for i in range(0, len(df), 384):
        sub_array = df[i * 384 : (i + 1) * 384, 1:].to_numpy()
        if i == 0:
            print(sub_array.shape)
        np.save(output_dir / f"{i:06d}.npy", sub_array)


def kami_split(cfg):
    # kami

    for path in cfg.exp.kami_pred_paths:
        base_name = path.replace("_submission.parquet", "").split("/")[-1]
        print(base_name)

        # valid
        valid_output_dir = Path(cfg.exp.output_dir) / base_name / "valid"
        valid_output_dir.mkdir(parents=True, exist_ok=True)
        valid_path = path.replace("_submission.parquet", "_valid_pred.parquet")
        df = pl.read_parquet(valid_path)
        array = df.select(TARGET_COLUMNS)[-cfg.exp.n_data_for_eval :].to_numpy()
        print(array.shape)
        for i in range(0, cfg.exp.n_data_for_eval, 384):
            sub_array = array[i * 384 : (i + 1) * 384]
            np.save(valid_output_dir / f"{i:06d}.npy", sub_array)

        # test
        test_output_dir = Path(cfg.exp.output_dir) / base_name / "test"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        sub_path = path
        df = pl.read_parquet(sub_path)
        array = df.select(TARGET_COLUMNS).to_numpy()
        print(array.shape)
        for i in range(0, len(df), 384):
            sub_array = array[i * 384 : (i + 1) * 384]
            np.save(test_output_dir / f"{i:06d}.npy", sub_array)


def kurupical_split(cfg):
    # kurupical

    for path in cfg.exp.kurupical_pred_paths:
        base_name = path.split("/")[-1]
        print(base_name)

        # valid
        valid_output_dir = Path(cfg.exp.output_dir) / base_name / "valid"
        valid_output_dir.mkdir(parents=True, exist_ok=True)
        valid_path = Path(path) / "pred_valid.parquet"
        df = pl.read_parquet(valid_path)
        array = df.select(TARGET_COLUMNS)[-cfg.exp.n_data_for_eval :].to_numpy()
        print(array.shape)
        for i in range(0, cfg.exp.n_data_for_eval, 384):
            sub_array = array[i * 384 : (i + 1) * 384]
            np.save(valid_output_dir / f"{i:06d}.npy", sub_array)

        # test
        test_output_dir = Path(cfg.exp.output_dir) / base_name / "test"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        sub_path = Path(path) / "submission.parquet"
        df = pl.read_parquet(sub_path)
        array = df.select(TARGET_COLUMNS).to_numpy()
        print(array.shape)
        for i in range(0, len(df), 384):
            sub_array = array[i * 384 : (i + 1) * 384]
            np.save(test_output_dir / f"{i:06d}.npy", sub_array)


def takoi_slit(cfg):
    # takoi
    for path in cfg.exp.takoi_pred_dir:
        base_name = path.replace("_pp.parquet", "").split("/")[-1]
        print(base_name)

        # valid
        valid_output_dir = Path(cfg.exp.output_dir) / base_name / "valid"
        valid_output_dir.mkdir(parents=True, exist_ok=True)
        valid_path = path.replace("_pp.parquet", "_val_preds.npy").replace("ex", "exp")
        array = np.load(valid_path)
        array = array[-cfg.exp.n_data_for_eval :]
        print(array.shape)
        for i in range(0, cfg.exp.n_data_for_eval, 384):
            sub_array = array[i * 384 : (i + 1) * 384]
            np.save(valid_output_dir / f"{i:06d}.npy", sub_array)

        # test
        test_output_dir = Path(cfg.exp.output_dir) / base_name / "test"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        sub_path = path
        df = pl.read_parquet(sub_path)
        array = df.select(TARGET_COLUMNS).to_numpy()
        print(array.shape)
        for i in range(0, len(df), 384):
            sub_array = array[i * 384 : (i + 1) * 384]
            np.save(test_output_dir / f"{i:06d}.npy", sub_array)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    base_split(cfg)
    kami_split(cfg)
    kurupical_split(cfg)
    takoi_slit(cfg)


if __name__ == "__main__":
    main()
