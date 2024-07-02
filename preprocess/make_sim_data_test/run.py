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
    test_df = pl.read_parquet(cfg.exp.test_data_path, n_rows=500 if cfg.debug else None)
    test_old_df = pl.read_parquet(
        cfg.exp.test_old_data_path, n_rows=500 if cfg.debug else None
    )

    save_dir = Path(cfg.exp.output_dir) / "test"
    save_dir.mkdir(exist_ok=True, parents=True)
    print(save_dir)

    base_array = test_df[:, 1:].to_numpy()
    old_array = test_old_df[:, 1:].to_numpy()

    df = (
        pl.DataFrame(base_array[:, :1])
        .with_row_index()
        .with_columns(
            [
                (pl.col("index") % 384).alias("location"),
                ((pl.col("index") // 384) * 24).alias("file_index"),
                ((pl.col("index") // 384) * 24 % 120).alias("time_mod"),
                ((pl.col("index") // 384) * 24 * 384 + (pl.col("index") % 384)).alias(
                    "row_index"
                ),
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
    data_old = (data_old - feat_mean_dict["base"]) / (feat_std_dict["base"] + 1e-60)

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

    df_similar = df.with_columns(
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

    for start_ri in tqdm(range(0, len(df_similar), 384)):
        end_ri = start_ri + 384

        original_x = base_array[start_ri:end_ri, :556]
        original_y = base_array[start_ri:end_ri, 556:]

        sim_x = [
            old_array[top_k_similar[start_ri:end_ri, i], :556]
            for i in range(cfg.exp.topk)
        ]

        np.savez(
            save_dir / f"id{start_ri}.npz",
            x=original_x,
            y=original_y,
            sim_x=sim_x,
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
