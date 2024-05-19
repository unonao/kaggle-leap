import os
import sys
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm


def get_tmelt_tice(cfg, df: pl.DataFrame):
    tmelt_list = []
    tice_list = []
    for h in range(60):
        tmp = df
        tmp = (
            tmp.with_columns(
                [
                    # 1200倍してスケールを揃えておく
                    (pl.col(f"ptend_t_{h}") * 1200.0),
                    (pl.col(f"ptend_q0001_{h}") * 1200.0),
                    (pl.col(f"ptend_q0002_{h}") * 1200.0),
                    (pl.col(f"ptend_q0003_{h}") * 1200.0),
                ]
            )
            .with_columns(
                [
                    (pl.col(f"state_q0002_{h}") + pl.col(f"state_q0003_{h}")).alias(
                        f"state_cloud_water_{h}"
                    ),
                    (pl.col(f"ptend_q0002_{h}") + pl.col(f"ptend_q0003_{h}")).alias(
                        f"ptend_cloud_water_{h}"
                    ),
                    (pl.col(f"state_t_{h}") + pl.col(f"ptend_t_{h}")).alias(
                        f"new_t_{h}"
                    ),
                    (pl.col(f"state_q0001_{h}") + pl.col(f"ptend_q0001_{h}")).alias(
                        f"new_q0001_{h}"
                    ),
                    (pl.col(f"state_q0002_{h}") + pl.col(f"ptend_q0002_{h}")).alias(
                        f"new_q0002_{h}"
                    ),
                    (pl.col(f"state_q0003_{h}") + pl.col(f"ptend_q0003_{h}")).alias(
                        f"new_q0003_{h}"
                    ),
                ]
            )
            .with_columns(
                [
                    (
                        pl.col(f"state_cloud_water_{h}")
                        + pl.col(f"ptend_cloud_water_{h}")
                    ).alias(f"new_cloud_water_{h}"),
                ]
            )
            .with_columns(
                [
                    # 絶対値が元の値の絶対値以下なら０埋めする
                    pl.when(
                        pl.col(f"new_cloud_water_{h}").abs()
                        < pl.col(f"ptend_cloud_water_{h}").abs() / 1200
                    )
                    .then(0)
                    .otherwise(pl.col(f"new_cloud_water_{h}"))
                    .alias(f"new_cloud_water_{h}"),
                    pl.when(
                        pl.col(f"new_q0002_{h}").abs()
                        < pl.col(f"ptend_q0002_{h}").abs() / 1200
                    )
                    .then(0)
                    .otherwise(pl.col(f"new_q0002_{h}"))
                    .alias(f"new_q0002_{h}"),
                    pl.when(
                        pl.col(f"new_q0003_{h}").abs()
                        < pl.col(f"ptend_q0003_{h}").abs() / 1200
                    )
                    .then(0)
                    .otherwise(pl.col(f"new_q0003_{h}"))
                    .alias(f"new_q0003_{h}"),
                ]
            )
        )

        cols = [
            f"state_t_{h}",
            f"new_cloud_water_{h}",
            f"new_q0002_{h}",
            f"new_q0003_{h}",
        ]

        zero_cld_water = len(tmp[cols].filter((pl.col(f"new_cloud_water_{h}") == 0)))
        zero_q2 = len(
            tmp[cols].filter(
                (pl.col(f"new_cloud_water_{h}") != 0) & (pl.col(f"new_q0002_{h}") == 0)
            )
        )
        zero_q3 = len(
            tmp[cols].filter(
                (pl.col(f"new_cloud_water_{h}") != 0) & (pl.col(f"new_q0003_{h}") == 0)
            )
        )
        df4 = tmp[cols].filter(
            (pl.col(f"new_cloud_water_{h}") != 0)
            & (pl.col(f"new_q0002_{h}") != 0)
            & (pl.col(f"new_q0003_{h}") != 0)
        )
        print(
            h,
            f"{zero_cld_water=} {zero_q2=} {zero_q3=}",
            df4.shape,
            df4[f"state_t_{h}"].min(),
            df4[f"state_t_{h}"].max(),
        )
        tmelt_list.append(df4[f"state_t_{h}"].max())
        tice_list.append(df4[f"state_t_{h}"].min())
    tmelt_array = np.array(tmelt_list).astype(np.float64)
    tice_array = np.array(tice_list).astype(np.float64)

    return tmelt_array, tice_array


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.preprocess_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    df = pl.read_parquet("input/train.parquet", n_rows=cfg.exp.num_rows)
    print(df.shape)

    tmelt_array, tice_array = get_tmelt_tice(cfg, df)

    # save
    np.save(output_path / "tmelt_array.npy", tmelt_array)
    np.save(output_path / "tice_array.npy", tice_array)


if __name__ == "__main__":
    main()
