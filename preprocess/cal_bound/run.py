import os
import pickle
import sys
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import xarray as xr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm.auto import tqdm


def cal_y_min_max(df):
    """
    7年分を3つに分割して約2年分にする
    また、それぞれを分布が偏らないように均等に4つに分割し、testの間隔に合わせるようにして最小と最大を求める
    """

    print("cal_y_min_max")
    y_sample_min_list = []
    y_sample_max_list = []

    splited_df = []
    n_rows = df.shape[0]
    for i in range(3):
        splited_df.append(df[i * n_rows // 3 : (i + 1) * n_rows // 3])

    dfs = []
    for i in range(3):
        for j in range(4):
            dfs.append(splited_df[i][j::4])

    for pseudo_test_df in dfs:
        y_sample_min = pseudo_test_df.min().to_numpy().ravel()
        y_sample_min_list.append(y_sample_min)
        y_sample_max = pseudo_test_df.max().to_numpy().ravel()
        y_sample_max_list.append(y_sample_max)

    y_sample_min_max = np.max(y_sample_min_list, axis=0)
    y_sample_max_min = np.min(y_sample_max_list, axis=0)
    return y_sample_min_max, y_sample_max_min


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.preprocess_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    df = pl.read_parquet("input/train.parquet", n_rows=50000 if cfg.debug else None)
    print(df.shape)

    y_sample_min_max, y_sample_max_min = cal_y_min_max(df)
    np.save(output_path / "y_sample_min_max.npy", y_sample_min_max)
    np.save(output_path / "y_sample_max_min.npy", y_sample_max_min)


if __name__ == "__main__":
    main()
