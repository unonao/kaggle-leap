from pathlib import Path

import numpy as np
import polars as pl

debug = False
df = pl.read_parquet("../input/train.parquet", n_rows=100 if debug else None)

output_path = Path("normalize_v2")
output_path.mkdir(exist_ok=True)



x_1_percentile = df[:, 1:557].quantile(0.01).to_numpy().ravel()
np.save(output_path / "x_1_percentile.npy", x_1_percentile)
x_99_percentile = df[:, 1:557].quantile(0.99).to_numpy().ravel()
np.save(output_path / "x_99_percentile.npy", x_99_percentile)

x_diff = x_99_percentile - x_1_percentile
x_lower_bound = x_1_percentile -  x_diff
np.save(output_path / "x_lower_bound.npy", x_lower_bound)

x_upper_bound = x_99_percentile + x_diff
np.save(output_path / "x_upper_bound.npy", x_upper_bound)

print(f"{x_1_percentile.shape=}, {x_99_percentile.shape=}, {x_lower_bound.shape=}, {x_upper_bound.shape=}")

x = df[:, 1:557].to_numpy()
# x_lower_bound 未満、x_upper_bound 以上の値を nan に変換して統計量の計算時に無視する
x[x<x_lower_bound] = np.nan
x[x>x_upper_bound] = np.nan
x_nanmean = np.nanmean(x, axis=0)
np.save(output_path / "x_nanmean.npy", x_nanmean)
x_nanmin = np.nanmin(x, axis=0)
np.save(output_path / "x_nanmin.npy", x_nanmin)
x_nanmax = np.nanmax(x, axis=0)
np.save(output_path / "x_nanmax.npy", x_nanmax)
x_nanstd = np.nanstd(x, axis=0)
np.save(output_path / "x_nanstd.npy", x_nanstd)

print(f"{x_nanmean.shape=}, {x_nanmin.shape=}, {x_nanmax.shape=}, {x_nanstd.shape=}")



### y
y_1_percentile = df[:, 557:].quantile(0.01).to_numpy().ravel()
np.save(output_path / "y_1_percentile.npy", y_1_percentile)
y_99_percentile = df[:, 557:].quantile(0.99).to_numpy().ravel()
np.save(output_path / "y_99_percentile.npy", y_99_percentile)

y_diff = y_99_percentile - y_1_percentile
y_lower_bound = y_1_percentile -  y_diff
np.save(output_path / "y_lower_bound.npy", y_lower_bound)

y_upper_bound = y_99_percentile + y_diff
np.save(output_path / "y_upper_bound.npy", y_upper_bound)

print(f"{y_1_percentile.shape=}, {y_99_percentile.shape=}, {y_lower_bound.shape=}, {y_upper_bound.shape=}")

y = df[:, 557:].to_numpy()
y[y<y_lower_bound] = np.nan
y[y>y_upper_bound] = np.nan
y_nanmean = np.nanmean(y, axis=0)
np.save(output_path / "y_nanmean.npy", y_nanmean)
y_nanmin = np.nanmin(y, axis=0)
np.save(output_path / "y_nanmin.npy", y_nanmin)
y_nanmax = np.nanmax(y, axis=0)
np.save(output_path / "y_nanmax.npy", y_nanmax)
y_nanstd = np.nanstd(y, axis=0)
np.save(output_path / "y_nanstd.npy", y_nanstd)

y_rms_np = np.sqrt(
    np.nanmean(y*y, axis=0)
).ravel()
np.save(output_path / "y_rms.npy", y_rms_np)

y_sub = y - y_nanmin
y_rms_sub_np = np.sqrt(np.nanmean(y_sub * y_sub, axis=0)).ravel()
np.save(output_path / "y_rms_sub.npy", y_rms_sub_np)
print(f"{y_nanmean.shape=}, {y_nanmin.shape=}, {y_nanmax.shape=}, {y_nanstd.shape=}, {y_rms_np.shape=}, {y_rms_sub_np.shape=}")


# 内容チェック
for i in [138]:
    print(f"{i=}, {x_nanmean[i]=}, {x_nanmin[i]=}, {x_nanmax[i]=}, {x_nanstd[i]=}")
for i in [193]:
    print(f"{i=}, {y_nanmean[i]=}, {y_nanmin[i]=}, {y_nanmax[i]=}, {y_nanstd[i]=}, {y_rms_np[i]=}, {y_rms_sub_np[i]=}")