import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.preprocess_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    train_df = pl.scan_parquet(Path(cfg.dir.input_dir) / "train.parquet")

    # 末尾の640,000だけ取り出し
    valid_df = train_df.tail(640_000).collect()
    print(valid_df)

    valid_df.write_parquet(output_path / "valid.parquet")


if __name__ == "__main__":
    main()
