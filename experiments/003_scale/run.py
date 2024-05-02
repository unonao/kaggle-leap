import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars as pl
import torch
import webdataset as wds
from google.cloud import storage
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from timm.utils import ModelEmaV2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import wandb
from utils.metric import score


class Scaler:
    def __init__(self, cfg):
        min_std = 1e-8

        self.x_mean = np.load(Path(cfg.exp.scale_dir) / "x_mean.npy")
        self.x_std = np.maximum(np.load(Path(cfg.exp.scale_dir) / "x_std.npy"), min_std)
        self.y_mean = np.load(Path(cfg.exp.scale_dir) / "y_mean.npy")
        self.y_rms_sub = np.maximum(
            np.load(Path(cfg.exp.scale_dir) / "y_rms_sub.npy"), min_std
        )

    def scale_input(self, x):
        return (x - self.x_mean) / self.x_std

    def scale_output(self, y):
        return (y - self.y_mean) / self.y_rms_sub

    def inv_scale_output(self, y):
        return y * self.y_rms_sub + self.y_mean

    def __call__(self, x, y):
        return self.scale_input(x), self.scale_output(y)


class LeapLightningDataModule(LightningDataModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.scaler = Scaler(cfg)
        self.cfg = cfg
        self.rng = random.Random(self.cfg.exp.seed)
        self.train_years = (1, 8) if cfg.debug is False else (1, 2)
        self.valid_years = (
            (1, 8) if cfg.debug is False else (1, 2)
        )  # (8, 10) if cfg.debug is False else (9, 10)
        self.train_dataset = self._make_dataset("train")
        self.valid_dataset = self._make_dataset("valid")

    class TestDataset(Dataset):
        def __init__(self, cfg, test_df, scaler):
            self.cfg = cfg
            self.scaler = scaler
            # 提供データは cam_in_SNOWHICE は削除済みなので削除しないが、idを削除する
            self.x = test_df[:, 1:].to_numpy()
            self.dtype = torch.float64 if "64" in cfg.exp.precision else torch.float32

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, index):
            return torch.from_numpy(self.scaler.scale_input(self.x[index])).to(
                self.dtype
            )

    def train_dataloader(self):
        return (
            wds.WebLoader(
                self.train_dataset,
                batch_size=None,
                num_workers=self.cfg.exp.num_workers,
            )
            .shuffle(7)
            .batched(
                batchsize=self.cfg.exp.train_batch_size,
                partial=False,
            )
        )

    def val_dataloader(self):
        return (
            wds.WebLoader(
                self.valid_dataset,
                batch_size=None,
                num_workers=self.cfg.exp.num_workers,
            )
            .shuffle(7)
            .batched(
                batchsize=self.cfg.exp.valid_batch_size,
                partial=False,
            )
        )

    def test_dataloader(self):
        self.test_df = pl.read_parquet(
            self.cfg.exp.test_path, n_rows=(None if self.cfg.debug is False else 500)
        )
        self.test_dataset = self.TestDataset(self.cfg, self.test_df, self.scaler)
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.exp.valid_batch_size * 384,
            num_workers=self.cfg.exp.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def _get_webdataset_url_list(self, bucket_name, prefix):
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=None)
        return sorted(
            [
                f"pipe:gsutil cat gs://{self.cfg.dir.gcs_bucket}/{path.name}"
                for path in blobs
                if path.name.endswith("tar")
            ]
        )

    def _make_tar_list(self, mode="train"):
        tar_list = []
        start_year, start_month = (
            self.cfg.exp.train_start if mode == "train" else self.cfg.exp.valid_start
        )
        end_year, end_month = (
            self.cfg.exp.train_end if mode == "train" else self.cfg.exp.valid_end
        )
        if self.cfg.debug:
            start_year, start_month = 1, 2
            end_year, end_month = 1, 2

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                if (year == start_year and month < start_month) or (
                    year == end_year and month > end_month
                ):
                    continue
                tmp = self._get_webdataset_url_list(
                    self.cfg.dir.gcs_bucket,
                    f"{self.cfg.dir.gcs_base_dir}/{self.cfg.exp.dataset_dir}/shards_{year:04d}-{month:02d}",
                )
                tar_list += tmp
        # 1/data_skip_mod の数にする
        if mode == "train" and self.cfg.exp.train_data_skip_mod:
            tar_list = tar_list[:: self.cfg.exp.train_data_skip_mod]
        elif mode == "valid" and self.cfg.exp.valid_data_skip_mod:
            tar_list = tar_list[:: self.cfg.exp.valid_data_skip_mod]

        print(mode, f"{len(tar_list)=}")
        return tar_list

    def _sum_dataset_sizes(self, bucket_name, prefix):
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
        total_size = 0
        for blob in blobs:
            if blob.name.endswith("dataset-size.json"):
                json_data = json.loads(blob.download_as_string())
                dataset_size = json_data["dataset size"]
                total_size += dataset_size
        return total_size

    def _get_dataset_size(self, mode="train"):
        start_year, start_month = (
            self.cfg.exp.train_start if mode == "train" else self.cfg.exp.valid_start
        )
        end_year, end_month = (
            self.cfg.exp.train_end if mode == "train" else self.cfg.exp.valid_end
        )
        if self.cfg.debug:
            start_year, start_month = 1, 2
            end_year, end_month = 1, 2

        total_size = 0
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                if (year == start_year and month < start_month) or (
                    year == end_year and month > end_month
                ):
                    continue
                tmp = self._sum_dataset_sizes(
                    self.cfg.dir.gcs_bucket,
                    f"{self.cfg.dir.gcs_base_dir}/{self.cfg.exp.dataset_dir}/shards_{year:04d}-{month:02d}",
                )
                total_size += tmp

        # 1/data_skip_mod の数にする
        if mode == "train" and self.cfg.exp.train_data_skip_mod:
            total_size = total_size // self.cfg.exp.train_data_skip_mod
        elif mode == "valid" and self.cfg.exp.valid_data_skip_mod:
            total_size = total_size // self.cfg.exp.valid_data_skip_mod
        # 1ファイルに約384ずつまとめているのでそれで割っておく
        total_size = total_size // 384
        return total_size

    def _make_dataset(self, mode="train"):
        tar_list = self._make_tar_list(mode)
        dataset_size = self._get_dataset_size(mode)
        dtype = torch.float64 if "64" in self.cfg.exp.precision else torch.float32
        print(mode, dataset_size)
        dataset = None
        if mode == "train":
            dataset = wds.WebDataset(urls=tar_list, shardshuffle=True).shuffle(
                100, rng=self.rng
            )
        else:
            dataset = wds.WebDataset(urls=tar_list, shardshuffle=False)

        return (
            dataset.decode()
            .to_tuple("input.npy", "output.npy")
            .map_tuple(
                lambda x: torch.tensor(
                    self.scaler.scale_input(
                        np.delete(x, 375, 1)  # cam_in_SNOWHICE は削除
                    ),
                    dtype=dtype,
                ),
                lambda y: torch.tensor(self.scaler.scale_output(y), dtype=dtype),
            )
            .with_length(dataset_size)
        )


class LeapModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        # Initialize the layers
        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  # Normalization layer
            layers.append(nn.LeakyReLU(inplace=True))  # Activation
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LeapLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = LeapModel(**cfg.exp.model)
        self.loss_fc = nn.MSELoss()  # Using MSE for regression
        self.model_ema = None
        if self.cfg.exp.ema.use_ema:
            print("Using EMA")
            self.model_ema = ModelEmaV2(self.model, self.cfg.exp.ema.decay)

    def training_step(self, batch, batch_idx):
        mode = "train"
        x, y = batch
        x = torch.flatten(x, start_dim=0, end_dim=1)
        y = torch.flatten(y, start_dim=0, end_dim=1)

        out = self.__pred(x, mode)
        loss = self.loss_fc(out, y)
        self.log(
            f"{mode}_loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        mode = "valid"
        x, y = batch
        x = torch.flatten(x, start_dim=0, end_dim=1)
        y = torch.flatten(y, start_dim=0, end_dim=1)
        out = self.__pred(x, mode)
        loss = self.loss_fc(out, y)
        self.log(
            f"{mode}_loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    def predict_step(self, batch, batch_idx):
        mode = "test"
        x = batch
        out = self.__pred(x, mode)
        return out

    def __pred(self, x, mode: str) -> torch.Tensor:
        if (mode == "valid" or mode == "test") and (self.model_ema is not None):
            out = self.model_ema.module(x)
        else:
            out = self.model(x)
        return out

    def on_after_backward(self):
        if self.model_ema is not None:
            self.model_ema.update(self.model)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.exp.lr)

        """
        scheduler = tc.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.exp.lr,
            steps_per_epoch=len(train_dataloader),
            epochs=cfg.exp.epochs + 1,
            pct_start=0.1,
        )
        """
        # 1epoch分をwarmupとするための記述
        num_warmup_steps = (
            math.ceil(self.trainer.max_steps / self.cfg.exp.epoch) * 1
            if self.cfg.exp.scheduler.use_one_epoch_warmup
            else 0
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=self.trainer.max_steps,
            num_warmup_steps=num_warmup_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def train(cfg: DictConfig, output_path: Path, pl_logger) -> None:
    monitor = f"{cfg.exp.monitor}"
    dm = LeapLightningDataModule(cfg)
    model = LeapLightningModule(cfg)
    checkpoint_cb = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        verbose=True,
        monitor=monitor,
        mode=cfg.exp.monitor_mode,
        save_top_k=1,
        save_last=False,
        enable_version_counter=False,
    )
    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = RichProgressBar()  # leave=True
    model_summary = RichModelSummary(max_depth=2)
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=cfg.exp.early_stopping_patience,
        mode=cfg.exp.monitor_mode,
    )
    if cfg.debug:
        cfg.exp.max_epochs = 2

    trainer = Trainer(
        default_root_dir=output_path,
        accelerator=cfg.exp.accelerator,
        precision=cfg.exp.precision,
        max_epochs=cfg.exp.max_epochs,
        max_steps=cfg.exp.max_epochs
        * len(dm.train_dataset)
        // cfg.exp.train_batch_size,
        gradient_clip_val=cfg.exp.gradient_clip_val,
        accumulate_grad_batches=cfg.exp.accumulate_grad_batches,
        logger=pl_logger,
        log_every_n_steps=1,
        limit_train_batches=None if cfg.debug is False else 2,
        limit_val_batches=None if cfg.debug is False else 2,
        # deterministic=True,
        callbacks=[
            checkpoint_cb,
            lr_monitor,
            progress_bar,
            model_summary,
            early_stopping,
        ],
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        # sync_batchnorm=True,
        check_val_every_n_epoch=cfg.exp.check_val_every_n_epoch,
    )
    trainer.fit(model, dm)

    # copy checkpoint_cb.best_model_path
    shutil.copy(
        checkpoint_cb.best_model_path,
        output_path / "checkpoints" / "best_model.ckpt",
    )


def predict_valid(cfg: DictConfig, output_path: Path) -> None:
    # TODO: チームを組むならvalidationデータセットを揃えて出力を保存する
    model_module = LeapLightningModule.load_from_checkpoint(
        output_path / "checkpoints" / "best_model.ckpt", cfg=cfg
    )
    if cfg.exp.ema.use_ema:
        model = model_module.model_ema.module
    model = model_module.model

    dm = LeapLightningDataModule(cfg)
    dataloader = dm.val_dataloader()

    preds = []
    labels = []
    model = model.to("cuda")
    model.eval()
    for x, y in tqdm(dataloader):
        x, y = x.to("cuda"), y.to("cuda")
        x = torch.flatten(x, start_dim=0, end_dim=1)
        y = torch.flatten(y, start_dim=0, end_dim=1)
        with torch.no_grad():
            out = model(x)
        preds.append(out.cpu())
        labels.append(y.cpu())

    preds = torch.cat(preds).numpy()
    preds = Scaler(cfg).inv_scale_output(preds)
    print(type(preds), preds.shape)
    labels = torch.cat(labels).numpy()

    predict_df = pd.DataFrame(preds, columns=[i for i in range(preds.shape[1])])
    predict_df["id"] = range(len(predict_df))
    label_df = pd.DataFrame(labels, columns=[i for i in range(labels.shape[1])])
    label_df["id"] = range(len(label_df))

    r2_score = score(label_df, predict_df, "id")
    print(f"{r2_score=}")
    wandb.log({"r2_score": r2_score})


def predict_test(cfg: DictConfig, output_path: Path) -> None:
    model_module = LeapLightningModule.load_from_checkpoint(
        output_path / "checkpoints" / "best_model.ckpt", cfg=cfg
    )
    if cfg.exp.ema.use_ema:
        model = model_module.model_ema.module
    model = model_module.model

    dm = LeapLightningDataModule(cfg)
    dataloader = dm.test_dataloader()
    preds = []
    model = model.to("cuda")
    model.eval()
    for x in tqdm(dataloader):
        x = x.to("cuda")
        # webdatasetとは違い、batchでの読み出しではないのでflattenは必要ない
        with torch.no_grad():
            out = model(x)
        preds.append(out.cpu())

    preds = torch.cat(preds).numpy()
    preds = Scaler(cfg).inv_scale_output(preds)
    print(type(preds), preds.shape)

    # load sample
    sample_submission_df = pl.read_parquet(
        cfg.exp.sample_submission_path, n_rows=(None if cfg.debug is False else 500)
    )

    preds *= sample_submission_df[:, 1:].to_numpy()
    sample_submission_df = pl.concat(
        [
            sample_submission_df.select("sample_id"),
            pl.from_numpy(preds, schema=sample_submission_df.columns[1:]),
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

    seed_everything(cfg.exp.seed)

    pl_logger = WandbLogger(
        name=exp_name,
        project="kaggle-leap",
        mode="disabled",  # if cfg.debug else None,
    )
    pl_logger.log_hyperparams(cfg)

    if "train" in cfg.exp.modes:
        train(cfg, output_path, pl_logger)
    if "valid" in cfg.exp.modes:
        predict_valid(cfg, output_path)
    if "test" in cfg.exp.modes:
        predict_test(cfg, output_path)


if __name__ == "__main__":
    main()
