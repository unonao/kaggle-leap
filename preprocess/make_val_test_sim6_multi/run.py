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


def make_sim_data_from_df(cfg, df, old_df, save_dir, mode):
    # 30+120n に絞る
    old_filter_df = (
        old_df.with_row_index()
        .with_columns(
            [
                ((pl.col("index") // 384) * 30 % 120).alias("time_mod"),
            ]
        )
        .filter((pl.col("time_mod") - 30) % 120 == 0)
        .drop(["index", "time_mod"])
    )

    base_array = df[:, 1:].to_numpy()
    old_array = old_filter_df[:, 1:].to_numpy()

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

    if mode == "valid":
        stats_df = (
            (
                pl.DataFrame(base_array[:, :1])
                .with_row_index()
                .with_columns(
                    [
                        ((pl.col("index") // 384) * 24 % 120).alias("time_mod"),
                        (
                            (pl.col("index") // 384) * 24 * 384
                            + (pl.col("index") % 384)
                        ).alias("row_index"),
                    ]
                )
                .drop(["column_0"])
            )
            .with_columns(
                [
                    pl.Series(
                        "old_row_index",
                        values=(((top_k_similar.numpy() // 384) * 4 * 30 + 30) * 384)
                        + (top_k_similar.numpy() % 384),
                    ),
                ]
            )
            .with_columns(
                [
                    pl.col("old_row_index")
                    .list.slice(0, 1)
                    .list.contains(pl.col("row_index") + 384 * 6)
                    .alias("is_top1_next")
                ]
            )
            .with_columns(
                [
                    pl.col("old_row_index")
                    .list.slice(k - 1, 1)
                    .list.contains(pl.col("row_index") + 384 * i)
                    .alias(f"is_top{k}_{i}")
                    for i in [-18, -42, -66, -90, -114, 6, 30, 54, 78, 102]
                    for k in [1, 2, 3]
                ]
            )
        )
        all_size = len(stats_df)
        max_get_size = len(stats_df.filter(pl.col("time_mod") == 24))
        get_size = stats_df[["is_top1_next"]].sum().to_numpy()[0, 0]
        recall = get_size / max_get_size
        print(f"{all_size=}, {max_get_size=}, {get_size=}, {recall=}")

    for start_ri in tqdm(range(0, len(base_array), 384)):
        end_ri = start_ri + 384

        if mode == "valid":
            original_x = base_array[start_ri:end_ri, :556]
            original_y = base_array[start_ri:end_ri, 556:]
            top1 = old_array[top_k_similar[start_ri:end_ri, 0], :556]
            top2 = old_array[top_k_similar[start_ri:end_ri, 1], :556]
            top3 = old_array[top_k_similar[start_ri:end_ri, 2], :556]
            is_next = stats_df[start_ri:end_ri]["is_top1_next"].to_numpy()

            # version1
            is_in_bools = np.zeros((384, 5), dtype=bool)
            is_in_bools[:, :4] = stats_df[start_ri:end_ri][
                [f"is_top1_{i}" for i in [6, -18, -66, 78]]
            ]
            is_in_bools[np.sum(is_in_bools, axis=1) == 0, -1] = 1
            y_class = np.argmax(is_in_bools, axis=1)

            # version2
            is_in_bools = np.zeros((384, 11), dtype=bool)
            is_in_bools[:, :10] = stats_df[start_ri:end_ri][
                [f"is_top1_{i}" for i in [6, -18, -42, -66, -90, -114, 30, 54, 78, 102]]
            ]
            is_in_bools[np.sum(is_in_bools, axis=1) == 0, -1] = 1
            y_class11 = np.argmax(is_in_bools, axis=1)

            is_in_bools = np.zeros((384, 11), dtype=bool)
            is_in_bools[:, :10] = stats_df[start_ri:end_ri][
                [f"is_top2_{i}" for i in [6, -18, -42, -66, -90, -114, 30, 54, 78, 102]]
            ]
            is_in_bools[np.sum(is_in_bools, axis=1) == 0, -1] = 1
            y_class11_top2 = np.argmax(is_in_bools, axis=1)

            is_in_bools = np.zeros((384, 11), dtype=bool)
            is_in_bools[:, :10] = stats_df[start_ri:end_ri][
                [f"is_top3_{i}" for i in [6, -18, -42, -66, -90, -114, 30, 54, 78, 102]]
            ]
            is_in_bools[np.sum(is_in_bools, axis=1) == 0, -1] = 1
            y_class11_top3 = np.argmax(is_in_bools, axis=1)

            np.savez(
                save_dir / f"id{start_ri}.npz",
                x=original_x,
                y=original_y,
                is_next=is_next,
                y_class=y_class,
                y_class11=y_class11,
                y_class11_top2=y_class11_top2,
                y_class11_top3=y_class11_top3,
                top1=top1,
                top2=top2,
                top3=top3,
            )
        elif mode == "test":
            original_x = base_array[start_ri:end_ri, :556]
            top1 = old_array[top_k_similar[start_ri:end_ri, 0], :556]
            top2 = old_array[top_k_similar[start_ri:end_ri, 1], :556]
            top3 = old_array[top_k_similar[start_ri:end_ri, 2], :556]
            np.savez(
                save_dir / f"id{start_ri}.npz",
                x=original_x,
                top1=top1,
                top2=top2,
                top3=top3,
            )


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
    # valid data
    month_dirs = get_two_years_month_dirs(cfg.exp.valid_year_id)
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
    start_id = 0
    df = pl.DataFrame(
        np.vstack([np.load(path) for path in path_list[start_id::24]])
    ).with_row_index("sample_id")
    old_df = pl.DataFrame(
        np.vstack([np.load(path) for path in path_list[start_id::30]])
    ).with_row_index("sample_id")
    save_dir = Path(cfg.exp.output_dir) / "valid"
    save_dir.mkdir(exist_ok=True, parents=True)
    print(save_dir)
    make_sim_data_from_df(cfg, df, old_df, save_dir, mode="valid")

    # test data
    print("*" * 10)
    df = pl.read_parquet(cfg.exp.test_data_path, n_rows=500 if cfg.debug else None)
    old_df = pl.read_parquet(
        cfg.exp.test_old_data_path, n_rows=5000 if cfg.debug else None
    )
    save_dir = Path(cfg.exp.output_dir) / "test"
    save_dir.mkdir(exist_ok=True, parents=True)
    print(save_dir)
    make_sim_data_from_df(cfg, df, old_df, save_dir, mode="test")


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
