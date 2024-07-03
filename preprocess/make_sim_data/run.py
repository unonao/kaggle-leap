import json
import os
import pickle
import shutil
import sys
import time
from glob import glob
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import polars as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def compute_cosine_similarity(tensor1, tensor2):
    return torch.mm(tensor1, tensor2.t())


def get_top_k_similar_rows(matrix, target_matrix, k=5, chunk_size=1000, device="cpu"):
    matrix = matrix.to(device)
    normalized_matrix = normalize(matrix, p=2, dim=1)
    target_matrix = target_matrix.to(device)
    normalized_target_matrix = normalize(target_matrix, p=2, dim=1)

    top_k_indices = torch.empty((matrix.size(0), k), dtype=torch.long, device="cpu")

    dataset = TensorDataset(normalized_matrix)
    dataloader = DataLoader(dataset, batch_size=chunk_size)

    for ci, chunk in enumerate(tqdm(dataloader)):
        chunk_tensor = chunk[0]
        chunk_size_actual = chunk_tensor.size(0)

        cosine_similarities = compute_cosine_similarity(
            chunk_tensor, normalized_target_matrix
        ).to(device)
        top_k = torch.topk(cosine_similarities, k, dim=1)
        top_k_indices[ci * chunk_size : ci * chunk_size + chunk_size_actual] = (
            top_k.indices.cpu()
        )

    return top_k_indices


def get_two_years_month_dirs(year_id):
    start = 1 + year_id * 2
    month_dirs = (
        [f"train/000{start}-{str(m).zfill(2)}" for m in range(2, 13)]
        + [
            f"train/000{y}-{str(m).zfill(2)}"
            for y in range(start + 1, start + 2)
            for m in range(1, 13)
        ]
        + [f"train/000{start+2}-01"]
    )
    return month_dirs


def make_sim_data(cfg: DictConfig) -> None:
    for year_id in cfg.exp.year_index:
        # ベースとなるデータファイルのパスを取得
        month_dirs = get_two_years_month_dirs(year_id)
        path_list = sorted(
            [
                path
                for paths in [
                    glob(str(Path(cfg.exp.data_dir) / dir_path / "*"))
                    for dir_path in month_dirs
                ]
                for path in paths
            ]
        )

        if cfg.debug:
            path_list = path_list[:100]

        # 開始地点ごとに処理を行う(24ごとに作成されるので、最大24通り)
        for start_id in cfg.exp.start_index:
            save_dir = Path(cfg.exp.output_dir) / f"year{year_id}_start{start_id}"
            save_dir.mkdir(exist_ok=True, parents=True)
            print(save_dir)

            base_array = np.vstack([np.load(path) for path in path_list[start_id::24]])
            old_array = np.vstack([np.load(path) for path in path_list[start_id::30]])

            df = (
                pl.DataFrame(base_array[:, :1])
                .with_row_index()
                .with_columns(
                    [
                        (pl.col("index") % 384).alias("location"),
                        ((pl.col("index") // 384) * 24).alias("file_index"),
                        ((pl.col("index") // 384) * 24 % 120).alias("time_mod"),
                        (
                            (pl.col("index") // 384) * 24 * 384
                            + (pl.col("index") % 384)
                        ).alias("row_index"),
                    ]
                )
                .drop(["column_0"])
            )

            # scaling
            feat_mean_dict = pickle.load(
                open(
                    Path(cfg.exp.scale_dir) / "x_mean_feat_dict.pkl",
                    "rb",
                )
            )
            feat_std_dict = pickle.load(
                open(
                    Path(cfg.exp.scale_dir) / "x_std_feat_dict.pkl",
                    "rb",
                )
            )
            data = torch.tensor(base_array[:, :556])
            data_old = torch.tensor(old_array[:, :556])
            data = (data - feat_mean_dict["base"]) / (feat_std_dict["base"] + 1e-60)
            data_old = (data_old - feat_mean_dict["base"]) / (
                feat_std_dict["base"] + 1e-60
            )

            # flaot32
            data = data.float()
            data_old = data_old.float()

            # similar
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if cfg.debug:
                device = "cpu"
            top_k_similar = get_top_k_similar_rows(
                data,
                data_old,
                k=cfg.exp.topk,
                chunk_size=cfg.exp.chunk_size,
                device=device,
            )

            df_similar = (
                df.with_columns(
                    [
                        pl.Series(
                            "old_index",
                            values=(top_k_similar.numpy()),
                        ),
                        pl.Series(
                            "old_file_index",
                            values=(top_k_similar.numpy() // 384) * 30,
                        ),
                        pl.Series(
                            "old_row_index",
                            values=(top_k_similar.numpy() // 384 * 30 * 384)
                            + (top_k_similar.numpy() % 384),
                        ),
                    ]
                )
                .with_columns(
                    [
                        pl.col("old_row_index")
                        .list.contains(pl.col("row_index") + 384 * i)
                        .alias(f"is_in_next{i}")
                        for i in [-12, -6, 0, 6, 12]
                    ]
                )
                .with_columns(
                    [
                        pl.col("old_row_index")
                        .list.slice(k - 1, 1)
                        .list.contains(pl.col("row_index") + 384 * i)
                        .alias(f"is_in_top{k}_next{i}")
                        for i in [-12, -6, 0, 6, 12]
                        for k in [1, 2, 3]
                    ]
                )
            )
            print(
                df_similar[
                    [
                        f"is_in_top{k}_next{i}"
                        for i in [-12, -6, 0, 6, 12]
                        for k in [1, 2, 3]
                    ]
                ].sum()
            )
            print(
                "is_in_next6:",
                df_similar.filter(pl.col("time_mod") == 24)["is_in_next6"].sum(),
                len(df_similar.filter(pl.col("time_mod") == 24)),
            )
            print(
                "is_in_next12:",
                df_similar.filter(pl.col("time_mod") == 48)["is_in_next12"].sum(),
                len(df_similar.filter(pl.col("time_mod") == 48)),
            )

            for start_ri in tqdm(range(0, len(df_similar), 384)):
                end_ri = start_ri + 384

                original_x = base_array[start_ri:end_ri, :556]
                original_y = base_array[start_ri:end_ri, 556:]

                sim_x = [
                    old_array[top_k_similar[start_ri:end_ri, i], :556]
                    for i in range(cfg.exp.topk)
                ]

                is_in_bools = df_similar[start_ri:end_ri][
                    [f"is_in_next{i}" for i in [-12, -6, 0, 6, 12]]
                ]
                is_in_bools_each = [
                    df_similar[start_ri:end_ri][
                        [f"is_in_top{k}_next{i}" for i in [-12, -6, 0, 6, 12]]
                    ]
                    for k in [1, 2, 3]
                ]

                np.savez(
                    save_dir / f"id{start_ri}.npz",
                    x=original_x,
                    y=original_y,
                    sim_x=sim_x,
                    is_in_bools=is_in_bools,
                    is_in_bools_each=is_in_bools_each,
                )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.preprocess_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    make_sim_data(cfg)


if __name__ == "__main__":
    main()
