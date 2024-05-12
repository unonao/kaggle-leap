"""
y のoutlier を除外してから各統計量を計算する
"""

from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr
from tqdm import tqdm

debug = False
n_sampling = 9000 if debug else int(625000 * 0.8)
iter_sampling = 10 if debug else 100

output_path = Path("normalize_v5_feat")
output_path.mkdir(exist_ok=True)


def cal_y_min_max(df):
    print("cal_y_min_max")
    y_sample_min_list = []
    y_sample_max_list = []
    for i in tqdm(range(iter_sampling)):
        sample_df = df[:, 557:].sample(n_sampling, seed=i)
        y_sample_min = sample_df.min().to_numpy().ravel()
        y_sample_min_list.append(y_sample_min)
        y_sample_max = sample_df.max().to_numpy().ravel()
        y_sample_max_list.append(y_sample_max)

    y_sample_min_max = np.max(y_sample_min_list, axis=0)
    np.save(output_path / "y_sample_min_max.npy", y_sample_min)
    y_sample_max_min = np.min(y_sample_max_list, axis=0)
    np.save(output_path / "y_sample_max_min.npy", y_sample_max)
    return y_sample_min_max, y_sample_max_min


def cal_pressures(df):
    grid_path = "/kaggle/working/misc/grid_info/ClimSim_low-res_grid-info.nc"
    grid_info = xr.open_dataset(grid_path)
    hyai = grid_info["hyai"].to_numpy()
    hybi = grid_info["hybi"].to_numpy()
    p0 = 1e5
    ps = df["state_ps"].to_numpy()
    pressures_array = hyai * p0 + hybi[None, :] * ps[:, None]
    pressures_array = np.diff(pressures_array, n=1)
    print(f"{pressures_array.shape=}")
    return pressures_array


def cal_water(df):
    sq0001_cols = [f"state_q0001_{i}" for i in range(60)]
    sq0001_array = df[sq0001_cols].to_numpy()

    sq0002_cols = [f"state_q0002_{i}" for i in range(60)]
    sq0002_array = df[sq0002_cols].to_numpy()

    sq0003_cols = [f"state_q0003_{i}" for i in range(60)]
    sq0003_array = df[sq0003_cols].to_numpy()

    return sq0001_array + sq0002_array + sq0003_array


def cal_save_stats(array, name):
    x_mean = array.mean(axis=0)
    np.save(output_path / f"x_mean_{name}_{diff_rate}.npy", x_mean)
    x_min = array.min(axis=0)
    np.save(output_path / f"x_min_{name}_{diff_rate}.npy", x_min)

    x_max = array.max(axis=0)
    np.save(output_path / f"x_max_{name}_{diff_rate}.npy", x_max)

    x_std = array.std(axis=0)
    np.save(output_path / f"x_std_{name}_{diff_rate}.npy", x_std)
    print(f"{x_mean.shape}, {x_min.shape}, {x_max.shape}, {x_std.shape}")


def cal_stats_x_y(
    df, y_sample_min_max, y_sample_max_min, diff_rate, pressures, water, energy
):
    print(f"cal_stats_x_y: {diff_rate=}")

    y_diff = (y_sample_max_min - y_sample_min_max) * diff_rate
    y_lower_bound = y_sample_min_max - y_diff
    np.save(output_path / f"y_lower_bound_{diff_rate}.npy", y_lower_bound)

    y_upper_bound = y_sample_max_min + y_diff
    np.save(output_path / f"y_upper_bound_{diff_rate}.npy", y_upper_bound)

    y = df[:, 557:].to_numpy()
    y[y < y_lower_bound] = np.nan
    y[y > y_upper_bound] = np.nan
    y_nanmean = np.nanmean(y, axis=0)
    np.save(output_path / f"y_nanmean_{diff_rate}.npy", y_nanmean)
    y_nanmin = np.nanmin(y, axis=0)
    np.save(output_path / f"y_nanmin_{diff_rate}.npy", y_nanmin)
    y_nanmax = np.nanmax(y, axis=0)
    np.save(output_path / f"y_nanmax_{diff_rate}.npy", y_nanmax)
    y_nanstd = np.nanstd(y, axis=0)
    np.save(output_path / f"y_nanstd_{diff_rate}.npy", y_nanstd)

    y_rms_np = np.sqrt(np.nanmean(y * y, axis=0)).ravel()
    np.save(output_path / f"y_rms_{diff_rate}.npy", y_rms_np)

    y_sub = y - y_nanmean
    y_rms_sub_np = np.sqrt(np.nanmean(y_sub * y_sub, axis=0)).ravel()
    np.save(output_path / f"y_rms_sub_{diff_rate}.npy", y_rms_sub_np)
    print(
        f"{y_nanmean.shape=}, {y_nanmin.shape=}, {y_nanmax.shape=}, {y_nanstd.shape=}, {y_rms_np.shape=}, {y_rms_sub_np.shape=}"
    )

    print("start x")
    ### x
    # y で outlier とならない範囲のデータを抽出
    eps = 1e-60
    filter_bool = np.all(
        df[:, 557:].to_numpy() >= y_lower_bound - eps, axis=1
    ) & np.all(df[:, 557:].to_numpy() <= y_upper_bound + eps, axis=1)
    print(f"{df.shape=}")
    filter_df = df.filter(filter_bool)
    print(f"{df.shape=}, {filter_df.shape=}")

    base_array = filter_df[:, 1:557].to_numpy()
    cal_save_stats(base_array, "base")

    pressures_array = pressures[filter_bool]
    cal_save_stats(pressures_array, "pressures")

    water_array = water[filter_bool]
    cal_save_stats(water_array, "water")

    energy_array = energy[filter_bool]
    cal_save_stats(energy_array, "energy")


if __name__ == "__main__":
    df = pl.read_parquet("../input/train.parquet", n_rows=10000 if debug else None)

    y_sample_min_max, y_sample_max_min = cal_y_min_max(df)

    pressures = cal_pressures(df)
    water = cal_water(df)
    energy = water * pressures
    for diff_rate in [16.0, 4.0, 1.0, 0.5, 0.1, 0]:
        cal_stats_x_y(
            df, y_sample_min_max, y_sample_max_min, diff_rate, pressures, water, energy
        )
