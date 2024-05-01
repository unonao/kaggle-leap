from pathlib import Path

import numpy as np
import polars as pl

debug = False
df = pl.read_parquet("../input/train.parquet", n_rows=100 if debug else None)

output_path = Path("normalize")
output_path.mkdir(exist_ok=True)


x_mean_np = df[:, 1:557].mean().to_numpy().ravel()
np.save(output_path / "x_mean.npy", x_mean_np)
x_min_np = df[:, 1:557].min().to_numpy().ravel()
np.save(output_path / "x_min.npy", x_min_np)
x_max_np = df[:, 1:557].max().to_numpy().ravel()
np.save(output_path / "x_max.npy", x_max_np)
x_std_np = df[:, 1:557].std().to_numpy().ravel()
np.save(output_path / "x_std.npy", x_std_np)
print(f"{x_mean_np.shape}, {x_min_np.shape}, {x_max_np.shape}, {x_std_np.shape}")

y_mean_np = df[:, 557:].mean().to_numpy().ravel()
np.save(output_path / "y_mean.npy", y_mean_np)
y_min_np = df[:, 557:].min().to_numpy().ravel()
np.save(output_path / "y_min.npy", y_min_np)
y_max_np = df[:, 557:].max().to_numpy().ravel()
np.save(output_path / "y_max.npy", y_max_np)
y_std_np = df[:, 557:].std().to_numpy().ravel()
np.save(output_path / "y_std.npy", y_std_np)
y_rms_np = np.sqrt(
    np.mean(df[:, 557:].to_numpy() * df[:, 557:].to_numpy(), axis=0)
).ravel()
np.save(output_path / "y_rms.npy", y_rms_np)

y_sub = df[:, 557:].to_numpy() - y_mean_np
y_rms_sub_np = np.sqrt(np.mean(y_sub * y_sub, axis=0)).ravel()
np.save(output_path / "y_rms_sub.npy", y_rms_sub_np)
print(
    f"{y_mean_np.shape}, {y_min_np.shape}, {y_max_np.shape}, {y_std_np.shape}, {y_rms_np.shape}",
    f"{y_rms_sub_np.shape}",
)
