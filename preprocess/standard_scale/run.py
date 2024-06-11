import os
import pickle
import sys
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.preprocess_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    df = pl.read_parquet("input/train.parquet", n_rows=5000 if cfg.debug else None)
    print(df.shape)

    y = df[:, 557:].to_numpy()

    # standard scale
    scaler = StandardScaler()
    scaler.fit(y)
    print(scaler.scale_[60:120])

    # save scaler
    with open(output_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()
