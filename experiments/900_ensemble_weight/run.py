import gc
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars as pl
from google.cloud import storage
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import utils
from utils.metric import score

N_ROWS_WHEN_DEBUG = 1000


def weighted_ensemble(preds, weights):
    assert len(preds) == len(weights)
    assert all([len(preds[0]) == len(pred) for pred in preds])

    ensemble_pred = np.zeros_like(preds[0])
    for pred, weight in zip(preds, weights):
        ensemble_pred += pred * weight

    return ensemble_pred


def ensemble(cfg, output_path):
    # check validation
    pred_list = []
    label_array = None
    for exp_name in cfg.exp.exp_names:
        gcs_path = (
            f"gs://{cfg.dir.gcs_bucket}/{cfg.dir.gcs_base_dir}/experiments/{exp_name}/"
        )
        print(f"exp_name: {exp_name}")
        if label_array is None:
            with utils.trace("load label"):
                label_array = pl.read_parquet(
                    gcs_path + "label.parquet",
                    retries=5,
                    n_rows=N_ROWS_WHEN_DEBUG if cfg.debug else None,
                )[:, 1:].to_numpy()
        with utils.trace("load pred"):
            pred = pl.read_parquet(
                gcs_path + "predict.parquet",
                retries=5,
                n_rows=N_ROWS_WHEN_DEBUG if cfg.debug else None,
            )[:, 1:].to_numpy()
        pred_list.append(pred)
        gc.collect()

    best_score = -1e40
    best_weights = None
    ss_df = pl.read_parquet(cfg.exp.sample_submission_path, n_rows=1)
    weight_array = ss_df.select(
        [x for x in ss_df.columns if x != "sample_id"]
    ).to_numpy()[0]
    for weights in cfg.exp.weights_list:
        ensemble_pred = weighted_ensemble(pred_list, weights)
        predict_weight_df = pd.DataFrame(
            ensemble_pred * weight_array,
            columns=[i for i in range(ensemble_pred.shape[1])],
        ).reset_index()
        label_weight_df = pd.DataFrame(
            label_array * weight_array,
            columns=[i for i in range(label_array.shape[1])],
        ).reset_index()

        r2_scores = score(
            label_weight_df,
            predict_weight_df,
            "index",
            multioutput="raw_values",
        )
        r2_score = np.mean(r2_scores)
        print(f"weights: {weights}, r2_score: {r2_score}")
        if r2_score > best_score:
            best_score = r2_score
            best_weights = weights

        del predict_weight_df, label_weight_df
        gc.collect()

    print(f"best_weights: {best_weights}, best_score: {best_score}")

    del pred_list
    gc.collect()

    # test ensemble
    pred_list = []
    for exp_name in cfg.exp.exp_names:
        gcs_path = (
            f"gs://{cfg.dir.gcs_bucket}/{cfg.dir.gcs_base_dir}/experiments/{exp_name}/"
        )
        with utils.trace("load pred"):
            pred = pl.read_csv(
                gcs_path + "submission.csv",
                n_rows=N_ROWS_WHEN_DEBUG if cfg.debug else None,
            )[:, 1:].to_numpy()
        pred_list.append(pred)
        gc.collect()

    sample_submission_df = pl.read_parquet(
        cfg.exp.sample_submission_path, n_rows=N_ROWS_WHEN_DEBUG if cfg.debug else None
    )
    ensemble_pred = weighted_ensemble(
        pred_list, best_weights
    )  # weightはすでにかかっているのでかけない
    sample_submission_df = pl.concat(
        [
            sample_submission_df.select("sample_id"),
            pl.from_numpy(ensemble_pred, schema=sample_submission_df.columns[1:]),
        ],
        how="horizontal",
    )
    sample_submission_df.write_csv(output_path / "submission.csv")
    print(sample_submission_df)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.exp_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    ensemble(cfg, output_path)


if __name__ == "__main__":
    main()
