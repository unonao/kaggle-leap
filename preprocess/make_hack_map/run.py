import datetime
import gc
import os
import sys
from glob import glob
from pathlib import Path

import hydra
import netCDF4  # implicitly required by xarray to load .nc
import numpy as np
import pandas as pd
import polars as pl
import tqdm
import xarray as xr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm


def make_hack_map(cfg, output_path):
    paths = sorted(glob("input/ClimSim_low-res/train/0002-*/*.npy"))
    pbuf_ozone_list = []
    month_list = []
    day_list = []
    seconds_list = []

    for i, path in enumerate(tqdm(paths)):
        x = np.load(path)
        pbuf_ozone_list.append(x[:, 376 : 376 + 10].copy())
        name = Path(path).name
        timestamp_str = name.split(".")[0]
        year, month, day, seconds = timestamp_str.split("-")
        year, month, day, seconds = int(year), int(month), int(day), int(seconds)
        month_list.append([month] * 384)
        day_list.append([day] * 384)
        seconds_list.append([seconds] * 384)

    pbuf_ozone_list = np.vstack(pbuf_ozone_list)
    month_list = [item for sublist in month_list for item in sublist]
    day_list = [item for sublist in day_list for item in sublist]
    seconds_list = [item for sublist in seconds_list for item in sublist]
    map_df = (
        pl.concat(
            [
                pl.from_numpy(
                    pbuf_ozone_list,
                    schema=[f"pbuf_ozone_{s}" for s in range(10)],
                ),
                pl.DataFrame(
                    [
                        pl.Series(name="month", values=month_list),
                        pl.Series(name="day", values=day_list),
                        pl.Series(name="seconds", values=seconds_list),
                    ]
                ),
            ],
            how="horizontal",
        )
        .with_columns((pl.col("seconds") // (20 * 60)).alias("tick"))
        .with_row_index("location")
        .with_columns(pl.col("location") % 384)
    ).drop(["timestamp_str", "year"])

    map_df.write_parquet(output_path / "map_hack.parquet")
    print(map_df.head())


def join_hack_map(cfg, output_path):
    print("load data")
    test_df = pl.read_parquet("input/test.parquet")
    test_old_df = pl.read_parquet("input/test_old.parquet")
    map_df = pl.read_parquet(output_path / "map_hack.parquet")

    print("join")
    ozeon_cols = [f"pbuf_ozone_{s}" for s in range(10)]
    test_info = (
        test_df[["sample_id"] + ozeon_cols]
        .with_columns([pl.col(col).round(12) for col in ozeon_cols])
        .join(
            map_df.with_columns([pl.col(col).round(12) for col in ozeon_cols]),
            on=ozeon_cols,
            how="left",
        )
        .drop(ozeon_cols)
    )
    print("null", test_info["month"].is_null().sum())
    print(test_info.head())

    test_old_info = (
        test_old_df[["sample_id", "pbuf_ozone_0"]]
        .join(map_df, on="pbuf_ozone_0", how="left")
        .drop(ozeon_cols)
    )
    print("null", test_old_info["month"].is_null().sum())
    print(test_old_info.head())

    test_info.write_parquet(output_path / "test_info.parquet")
    test_old_info.write_parquet(output_path / "test_old_info.parquet")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.preprocess_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    if "make" in cfg.exp.modes:
        make_hack_map(cfg, output_path)
    if "join" in cfg.exp.modes:
        join_hack_map(cfg, output_path)


if __name__ == "__main__":
    main()
