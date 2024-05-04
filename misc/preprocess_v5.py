"""
y のoutlier を除外してから各統計量を計算する
"""

from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

debug = False
n_sampling = 9000 if debug else int(625000 * 0.8)
iter_sampling = 10 if debug else 100

output_path = Path("normalize_v5")
output_path.mkdir(exist_ok=True)

df = pl.read_parquet("../input/train.parquet", n_rows=10000 if debug else None)

### y
print("start y")
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


for diff_rate in [0, 0.1]:
    print(f"{diff_rate=}")
    y_diff = (y_sample_max_min - y_sample_min_max) * diff_rate
    y_lower_bound = y_sample_min_max - y_diff
    np.save(output_path / f"y_lower_bound_{diff_rate}.npy", y_lower_bound)

    y_upper_bound = y_sample_max_min + y_diff
    np.save(output_path / f"y_upper_bound_{diff_rate}.npy", y_upper_bound)

    print(
        f"{y_sample_min.shape=}, {y_sample_max.shape=}, {y_lower_bound.shape=}, {y_upper_bound.shape=}"
    )

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
    df = df.filter(filter_bool)
    print(f"{df.shape=}")

    x_mean_np = df[:, 1:557].mean().to_numpy().ravel()
    np.save(output_path / f"x_mean_{diff_rate}.npy", x_mean_np)
    x_min_np = df[:, 1:557].min().to_numpy().ravel()
    np.save(output_path / f"x_min_{diff_rate}.npy", x_min_np)
    x_max_np = df[:, 1:557].max().to_numpy().ravel()
    np.save(output_path / f"x_max_{diff_rate}.npy", x_max_np)
    x_std_np = df[:, 1:557].std().to_numpy().ravel()
    np.save(output_path / f"x_std_{diff_rate}.npy", x_std_np)
    print(f"{x_mean_np.shape}, {x_min_np.shape}, {x_max_np.shape}, {x_std_np.shape}")

    # 内容チェック

    for i in [133, 193]:
        print(f" {i=}")
        print(f" {x_mean_np[i]=}, {x_min_np[i]=}, {x_max_np[i]=}, {x_std_np[i]=}")
        print(
            f" {y_sample_min_max[i]=}, {y_sample_max_min[i]=}, {y_lower_bound[i]=}, {y_upper_bound[i]=}"
        )
        print(
            f" {y_nanmean[i]=}, {y_nanmin[i]=}, {y_nanmax[i]=}, {y_nanstd[i]=}, {y_rms_np[i]=}, {y_rms_sub_np[i]=}"
        )
        print()
