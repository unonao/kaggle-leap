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

output_path = Path("normalize_v3")
output_path.mkdir(exist_ok=True)

df = pl.read_parquet("../input/train.parquet", n_rows=10000 if debug else None)

### y
print("start y")
y_001_percentile_list = []
y_999_percentile_list = []
for i in tqdm(range(iter_sampling)):
    sample_df = df[:, 557:].sample(n_sampling, seed=i)
    y_001_percentile = sample_df.quantile(0.001).to_numpy().ravel()
    y_001_percentile_list.append(y_001_percentile)
    y_999_percentile = sample_df.quantile(0.999).to_numpy().ravel()
    y_999_percentile_list.append(y_999_percentile)

y_001_percentile_max = np.max(y_001_percentile_list, axis=0)
np.save(output_path / "y_001_percentile_max.npy", y_001_percentile)
y_999_percentile_min = np.min(y_999_percentile_list, axis=0)
np.save(output_path / "y_999_percentile_min.npy", y_999_percentile)

y_diff = y_999_percentile_min - y_001_percentile_max
y_lower_bound = y_001_percentile_max - y_diff
np.save(output_path / "y_lower_bound.npy", y_lower_bound)

y_upper_bound = y_999_percentile_min + y_diff
np.save(output_path / "y_upper_bound.npy", y_upper_bound)

print(
    f"{y_001_percentile.shape=}, {y_999_percentile.shape=}, {y_lower_bound.shape=}, {y_upper_bound.shape=}"
)

y = df[:, 557:].to_numpy()
y[y < y_lower_bound] = np.nan
y[y > y_upper_bound] = np.nan
y_nanmean = np.nanmean(y, axis=0)
np.save(output_path / "y_nanmean.npy", y_nanmean)
y_nanmin = np.nanmin(y, axis=0)
np.save(output_path / "y_nanmin.npy", y_nanmin)
y_nanmax = np.nanmax(y, axis=0)
np.save(output_path / "y_nanmax.npy", y_nanmax)
y_nanstd = np.nanstd(y, axis=0)
np.save(output_path / "y_nanstd.npy", y_nanstd)

y_rms_np = np.sqrt(np.nanmean(y * y, axis=0)).ravel()
np.save(output_path / "y_rms.npy", y_rms_np)

y_sub = y - y_nanmean
y_rms_sub_np = np.sqrt(np.nanmean(y_sub * y_sub, axis=0)).ravel()
np.save(output_path / "y_rms_sub.npy", y_rms_sub_np)
print(
    f"{y_nanmean.shape=}, {y_nanmin.shape=}, {y_nanmax.shape=}, {y_nanstd.shape=}, {y_rms_np.shape=}, {y_rms_sub_np.shape=}"
)

print()

print("start x")
### x
# y で outlier とならない範囲のデータを抽出
eps = 1e-60
filter_bool = np.all(df[:, 557:].to_numpy() >= y_lower_bound - eps, axis=1) & np.all(
    df[:, 557:].to_numpy() <= y_upper_bound + eps, axis=1
)
print(f"{df.shape=}")
df = df.filter(filter_bool)
print(f"{df.shape=}")


x_mean_np = df[:, 1:557].mean().to_numpy().ravel()
np.save(output_path / "x_mean.npy", x_mean_np)
x_min_np = df[:, 1:557].min().to_numpy().ravel()
np.save(output_path / "x_min.npy", x_min_np)
x_max_np = df[:, 1:557].max().to_numpy().ravel()
np.save(output_path / "x_max.npy", x_max_np)
x_std_np = df[:, 1:557].std().to_numpy().ravel()
np.save(output_path / "x_std.npy", x_std_np)
print(f"{x_mean_np.shape}, {x_min_np.shape}, {x_max_np.shape}, {x_std_np.shape}")

# 内容チェック

for i in [133, 193]:
    print(f"{i=}")
    print(f"{x_mean_np[i]=}, {x_min_np[i]=}, {x_max_np[i]=}, {x_std_np[i]=}")
    print(
        f"{y_001_percentile_max[i]=}, {y_999_percentile_min[i]=}, {y_lower_bound[i]=}, {y_upper_bound[i]=}"
    )
    print(
        f"{y_nanmean[i]=}, {y_nanmin[i]=}, {y_nanmax[i]=}, {y_nanstd[i]=}, {y_rms_np[i]=}, {y_rms_sub_np[i]=}"
    )
    print()
