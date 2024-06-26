import gc
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import optuna
import pandas as pd
import polars as pl
from google.cloud import storage
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

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


def objective(trial, pred_list, label_array, weight_array):
    weights = [trial.suggest_float(f"weight_{i}", 0, 1) for i in range(len(pred_list))]
    # weightの合計を1にする
    weights = [w / sum(weights) for w in weights]
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
    return r2_score


def ensemble(cfg, output_path):
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
                    gcs_path + "val2_label.parquet",
                    retries=5,
                    n_rows=N_ROWS_WHEN_DEBUG if cfg.debug else None,
                )[:, 1:].to_numpy()
        with utils.trace("load pred"):
            pred = pl.read_parquet(
                gcs_path + "val2_predict.parquet",
                retries=5,
                n_rows=N_ROWS_WHEN_DEBUG if cfg.debug else None,
            )[:, 1:].to_numpy()
        pred_list.append(pred)

    ss_df = pl.read_parquet(cfg.exp.sample_submission_path, n_rows=1)
    weight_array = ss_df.select(
        [x for x in ss_df.columns if x != "sample_id"]
    ).to_numpy()[0]

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial,
            pred_list=pred_list,  # noqa: F821
            label_array=label_array,
            weight_array=weight_array,
        ),
        n_trials=cfg.exp.n_trials,
    )

    best_weights = [
        study.best_trial.params[f"weight_{i}"] for i in range(len(pred_list))
    ]
    best_weights = [w / sum(best_weights) for w in best_weights]
    best_score = study.best_value

    print(f"best_weights: {best_weights}, best_score: {best_score}")

    del pred_list
    gc.collect()
    for name in ["valid_pred", "submission"]:
        pred_list = []
        for exp_name in cfg.exp.exp_names:
            gcs_path = f"gs://{cfg.dir.gcs_bucket}/{cfg.dir.gcs_base_dir}/experiments/{exp_name}/"
            with utils.trace("load pred"):
                df = pl.read_parquet(
                    gcs_path + f"{name}.parquet",
                    n_rows=N_ROWS_WHEN_DEBUG if cfg.debug else None,
                )
                pred = df[:, 1:].to_numpy()
            pred_list.append(pred)
            gc.collect()

        ensemble_pred = weighted_ensemble(pred_list, best_weights)
        ensemble_df = pl.concat(
            [
                df.select("sample_id"),
                pl.from_numpy(ensemble_pred, schema=df.columns[1:]),
            ],
            how="horizontal",
        )
        ensemble_df.write_parquet(output_path / f"{name}.parquet")
        print(ensemble_df)
        del pred_list, ensemble_df
        gc.collect()


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
