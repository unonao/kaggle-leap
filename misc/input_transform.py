"""
y のoutlier を除外してから各統計量を計算する
"""

import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import PowerTransformer

debug = False

output_path = Path("input_transform")
output_path.mkdir(exist_ok=True)

print("loading data...")
df = pl.read_parquet("../input/train.parquet", n_rows=1000000)

print("mean")
x_mean_np = df[:, 1:557].mean().to_numpy().ravel()
np.save(output_path / "x_mean.npy", x_mean_np)

# yeo-johnson
print("yeo-johnson")
pt = PowerTransformer(method="yeo-johnson")
pt.fit(df[:, 1:557].to_numpy() - x_mean_np)

# save
with open(output_path / "yeo-johnson.pkl", "wb") as f:
    pickle.dump(pt, f)
