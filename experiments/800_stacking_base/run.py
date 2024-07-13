import gc
import glob
import itertools
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from adan import Adan
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
from sklearn.metrics import precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import utils
import wandb


def get_valid_name(n_fold, now_fold):
    return f"{n_fold}fold_{now_fold}fold"


class Scaler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.eps = cfg.exp.eps
        self.y_mean = np.load(Path(cfg.exp.scale_dir) / "y_nanmean.npy")
        self.y_rms_sub = np.maximum(
            np.load(Path(cfg.exp.scale_dir) / "y_rms_sub.npy"),
            self.eps,
        )

        self.tmelt_array = np.load(
            Path(cfg.exp.tmelt_tice_dir) / "tmelt_array.npy", allow_pickle=True
        )
        self.tice_array = np.load(
            Path(cfg.exp.tmelt_tice_dir) / "tice_array.npy",
            allow_pickle=True,
        )

        self.fill_target_index = [
            cfg.cols.col_names.index(col) for col in cfg.exp.fill_target
        ]

    def scale_input(self, x):
        x = (x - self.y_mean) / self.y_rms_sub
        return x

    def scale_output(self, y):
        y = (y - self.y_mean) / self.y_rms_sub
        return y

    def inv_scale_output(self, y, original_x, post_process=True):
        y = y * self.y_rms_sub + self.y_mean
        for i in range(self.y_rms_sub.shape[0]):
            if self.y_rms_sub[i] < self.eps * 1.1:
                y[:, i] = self.y_mean[i]

        if post_process:
            y = self.post_process(y, original_x)

        return y

    def post_process(self, y, original_x):
        # tmelt以上の値を置き換える
        tmelt_cond = original_x[:, :60] > self.tmelt_array
        y[:, 180:240] = np.where(
            tmelt_cond, original_x[:, 180:240] / (-1200), y[:, 180:240]
        )
        # tice以下の値を置き換える
        tmelt_cond = original_x[:, :60] < self.tice_array
        y[:, 120:180] = np.where(
            tmelt_cond, original_x[:, 120:180] / (-1200), y[:, 120:180]
        )

        # fill target はすべて置き換える
        y[:, self.fill_target_index] = original_x[:, self.fill_target_index] * (
            -1 / 1200
        )
        return y


class LeapLightningDataModule(LightningDataModule):
    def __init__(
        self,
        cfg,
        train_file_names,
        valid_file_names,
        test_file_names,
    ):
        super().__init__()
        self.scaler = Scaler(cfg)
        self.cfg = cfg
        self.train_file_names = train_file_names
        self.valid_file_names = valid_file_names
        self.test_file_names = test_file_names

    class TrainDataset(Dataset):
        def __init__(self, cfg, scaler, file_names):
            self.cfg = cfg
            self.scaler = scaler
            self.data_names = cfg.exp.data_names
            self.file_names = file_names

        def __len__(self):
            return len(self.file_names)

        def __getitem__(self, index):
            #
            original = np.load(
                Path(self.cfg.exp.dataset_dir)
                / "label"
                / f"valid/{self.file_names[index]}"
            )
            original_x, original_y = original[:556], original[556:]

            #  (384,368, k) だけのデータを読み込む (384はテストだともう少し少ないケースもあるので注意)
            data_list = []
            for data_name in self.data_names:
                data = np.load(
                    Path(self.cfg.exp.dataset_dir)
                    / data_name
                    / f"valid/{self.file_names[index]}"
                )
                data = self.scaler.scale_input(data)
                data_list.append(data)
            x = np.stack(data_list, axis=2)
            y = self.scaler.scale_output(original_y)

            return (
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(original_x),
                torch.from_numpy(original_y),
            )

    class TestDataset(Dataset):
        def __init__(self, cfg, scaler, file_names):
            self.cfg = cfg
            self.scaler = scaler
            self.data_names = cfg.exp.data_names
            self.file_names = file_names

        def __len__(self):
            return len(self.file_names)

        def __getitem__(self, index):
            #
            original_x = np.load(
                Path(self.cfg.exp.dataset_dir)
                / "label"
                / f"test/{self.file_names[index]}"
            )

            #  (384,368,k) だけのデータを読み込む (384はテストだともう少し少ないケースもあるので注意)
            data_list = []
            for data_name in self.data_names:
                data = np.load(
                    Path(self.cfg.exp.dataset_dir)
                    / data_name
                    / f"test/{self.file_names[index]}"
                )
                data = self.scaler.scale_input(data)
                data_list.append(data)
            x = np.stack(data_list, axis=2)

            return (
                torch.from_numpy(x),
                torch.from_numpy(original_x),
            )

    def train_dataloader(self):
        train_dataset = self.TrainDataset(
            self.cfg,
            self.scaler,
            self.train_file_names,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.cfg.exp.train_batch_size,
            num_workers=self.cfg.exp.num_workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self):
        valid_dataset = self.TrainDataset(
            self.cfg,
            self.scaler,
            self.valid_file_names,
        )
        return DataLoader(
            valid_dataset,
            batch_size=self.cfg.exp.valid_batch_size,
            num_workers=self.cfg.exp.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self):
        self.test_dataset = self.TestDataset(
            self.cfg, self.scaler, self.test_file_names
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.exp.valid_batch_size,
            num_workers=self.cfg.exp.num_workers,
            shuffle=False,
            pin_memory=False,
        )


class MLP(nn.Module):
    def __init__(self, in_size, hidden_sizes, use_layer_norm=False):
        super(MLP, self).__init__()
        layers = []
        previous_size = in_size
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_size, hidden_size))
            if i != len(hidden_sizes) - 1:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_size))
                layers.append(nn.LeakyReLU(inplace=True))
            previous_size = hidden_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class LeapModel(nn.Module):
    def __init__(self, hidden_dims=[256]):
        super().__init__()
        linear = []
        for hidden_dim in hidden_dims:
            linear.append(nn.Linear(hidden_dim, hidden_dim))
            linear.append(nn.LeakyReLU())
        linear.append(nn.Linear(hidden_dims[-1], 1))
        self.linear = nn.Sequential(*linear)

    def forward(self, x):
        return self.linear(x)


class LeapLightningModule(LightningModule):
    def __init__(self, cfg, fold):
        super().__init__()
        self.cfg = cfg
        self.model = LeapModel(
            **cfg.exp.model,
        )
        self.scaler = Scaler(cfg)
        self.loss_fc = nn.MSELoss()  # Using MSE for regression
        # self.loss_fc = nn.L1Loss()
        self.valid_name = get_valid_name(cfg.exp.n_fold, fold)
        self.torch_dtype = torch.float64 if "64" in cfg.exp.precision else torch.float32

        unuse_cols = list(itertools.chain.from_iterable(cfg.exp.unuse_cols_list))
        self.unuse_cols_index = [cfg.cols.col_names.index(col) for col in unuse_cols]
        self.use_cols_index = torch.tensor(
            [i for i in range(368) if i not in self.unuse_cols_index]
        )

        self.valid_preds = []
        self.valid_labels = []
        self.valid_original_xs = []

        ss_df = pl.read_parquet(cfg.exp.sample_submission_path, n_rows=1)
        self.weight_array = ss_df.select(
            [x for x in ss_df.columns if x != "sample_id"]
        ).to_numpy()[0]

    def training_step(self, batch, batch_idx):
        mode = "train"
        x, y, original_x, original_y = batch
        x, y = (
            torch.flatten(x, start_dim=0, end_dim=1),
            torch.flatten(y, start_dim=0, end_dim=1),
        )
        x, y = (
            x.to(self.torch_dtype),
            y.to(self.torch_dtype),
        )
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
        x, y, original_x, original_y = batch
        x, y = (
            torch.flatten(x, start_dim=0, end_dim=1),
            torch.flatten(y, start_dim=0, end_dim=1),
        )
        x, y = (
            x.to(self.torch_dtype),
            y.to(self.torch_dtype),
        )
        out = self.__pred(x, mode)
        loss = self.loss_fc(out, y)

        self.log(
            f"{mode}_loss/{self.valid_name}",
            loss.detach().item(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        self.valid_preds.append(out.cpu().to(torch.float64).numpy())
        self.valid_labels.append(original_y.cpu().to(torch.float64).numpy())
        self.valid_original_xs.append(original_x.cpu().to(torch.float64).numpy())
        return loss

    def predict_step(self, batch, batch_idx):
        mode = "test"
        x, x_cat, _ = batch
        x = x.to(self.torch_dtype)
        x_cat = x_cat.to(torch.long)
        out = self.__pred(x, x_cat, mode)
        return out

    def __pred(self, x, x_cat, x1, x1_cat, mode: str) -> torch.Tensor:
        if (mode == "valid" or mode == "test") and (self.model_ema is not None):
            out = self.model_ema.module(x, x_cat, x1, x1_cat)
        else:
            out = self.model(x, x_cat, x1, x1_cat)
        return out

    def on_after_backward(self):
        if self.model_ema is not None:
            self.model_ema.update(self.model)

    def on_validation_epoch_end(self):
        valid_preds = np.concatenate(self.valid_preds, axis=0).astype(np.float64)
        valid_labels = np.concatenate(self.valid_labels, axis=0).astype(np.float64)
        valid_original_xs = np.concatenate(self.valid_original_xs, axis=0).astype(
            np.float64
        )
        valid_preds = self.scaler.inv_scale_output(valid_preds, valid_original_xs)

        r2_scores = r2_score(
            valid_labels * self.weight_array,
            valid_preds * self.weight_array,
        )
        print(f"{r2_scores=}")
        self.log(
            f"valid_r2_score/{self.valid_name}",
            r2_scores,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.valid_preds = []
        self.valid_labels = []
        self.valid_original_xs = []
        gc.collect()

    def configure_optimizers(self):
        optimizer = None
        if self.cfg.exp.optimizer.name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.exp.optimizer.lr,
                weight_decay=self.cfg.exp.optimizer.weight_decay,
            )
        elif self.cfg.exp.optimizer.name == "RAdam":
            optimizer = torch.optim.RAdam(
                self.model.parameters(),
                lr=self.cfg.exp.optimizer.lr,
                weight_decay=self.cfg.exp.optimizer.weight_decay,
            )
        elif self.cfg.exp.optimizer.name == "Adan":
            optimizer = Adan(
                self.model.parameters(),
                lr=self.cfg.exp.optimizer.lr,
                weight_decay=self.cfg.exp.optimizer.weight_decay,
                betas=self.cfg.exp.optimizer.opt_betas,
                eps=self.cfg.exp.optimizer.eps,
                max_grad_norm=self.cfg.exp.optimizer.max_grad_norm,
                no_prox=self.cfg.exp.optimizer.no_prox,
            )

        if self.cfg.exp.scheduler.name == "CosineAnnealingWarmRestarts":
            # 1epoch分をwarmupとするための記述
            num_warmup_steps = (
                math.ceil(self.trainer.max_steps / self.cfg.exp.max_epochs) * 1
                if self.cfg.exp.scheduler.use_one_epoch_warmup
                else 0
            )
            if self.cfg.exp.val_check_interval:
                num_warmup_steps = min(num_warmup_steps, self.cfg.exp.val_check_warmup)
            print(f"{num_warmup_steps=}")
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=self.trainer.max_steps,
                num_warmup_steps=num_warmup_steps,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif self.cfg.exp.scheduler.name == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.cfg.exp.scheduler.base_lr,
                max_lr=self.cfg.exp.scheduler.max_lr,
                mode="triangular2",
                step_size_up=max(
                    int(
                        self.trainer.max_steps / self.cfg.exp.scheduler.num_cycles * 0.1
                    ),
                    1,
                ),
                step_size_down=int(
                    self.trainer.max_steps / self.cfg.exp.scheduler.num_cycles * 0.9
                ),
                cycle_momentum=False,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif self.cfg.exp.scheduler.name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.cfg.exp.scheduler.factor,
                patience=self.cfg.exp.scheduler.patience,
                threshold=self.cfg.exp.scheduler.threshold,
                threshold_mode=self.cfg.exp.scheduler.threshold_mode,
                cooldown=self.cfg.exp.scheduler.cooldown,
                min_lr=self.cfg.exp.scheduler.min_lr,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "frequency": self.cfg.exp.val_check_interval
                    if self.cfg.exp.val_check_interval
                    else 1,
                    "monitor": f"valid_loss/{self.valid_name}",
                    "interval": "step" if self.cfg.exp.val_check_interval else "epoch",
                },
            }


def train(cfg: DictConfig, output_path: Path, pl_logger) -> None:
    file_names = [
        Path(f).name
        for f in sorted(
            glob.glob(
                str(Path(cfg.exp.dataset_dir) / cfg.exp.data_names[0] / "valid/*.npy")
            )
        )
    ]
    test_file_names = [
        Path(f).name
        for f in sorted(
            glob.glob(
                str(Path(cfg.exp.dataset_dir) / cfg.exp.data_names[0] / "test/*.npy")
            )
        )
    ]

    # cross validation
    folds = KFold(n_splits=cfg.exp.n_fold, shuffle=True, random_state=cfg.exp.seed)
    for fold_id, (train_index, valid_index) in enumerate(folds.split(file_names)):
        train_file_names = [file_names[i] for i in train_index]
        valid_file_names = [file_names[i] for i in valid_index]
        print(
            f"{len(train_file_names)=}, {len(valid_file_names)=}, {len(test_file_names)=}"
        )
        monitor = f"valid_r2_score/{get_valid_name(cfg.exp.n_fold, fold_id)}"
        dm = LeapLightningDataModule(
            cfg, train_file_names, valid_file_names, test_file_names
        )
        model = LeapLightningModule(cfg, fold_id)
        checkpoint_cb = ModelCheckpoint(
            dirpath=output_path / f"checkpoints_{fold_id}",
            verbose=True,
            monitor=monitor,
            mode="max",
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
            mode="max",
        )
        if cfg.debug:
            cfg.exp.max_epochs = 2
        trainer = Trainer(
            default_root_dir=output_path,
            accelerator=cfg.exp.accelerator,
            precision=cfg.exp.precision,
            max_epochs=cfg.exp.max_epochs,
            max_steps=cfg.exp.max_epochs
            * len(dm.train_file_names)
            // cfg.exp.train_batch_size
            // cfg.exp.accumulate_grad_batches,
            gradient_clip_val=cfg.exp.gradient_clip_val,
            accumulate_grad_batches=cfg.exp.accumulate_grad_batches,
            logger=pl_logger,
            log_every_n_steps=1,
            limit_train_batches=None if cfg.debug is False else 4,
            limit_val_batches=None if cfg.debug is False else 4,
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
            val_check_interval=cfg.exp.val_check_interval,
        )

        trainer.fit(model, dm, ckpt_path=cfg.exp.resume_ckpt_path)

        # copy checkpoint_cb.best_model_path
        shutil.copy(
            checkpoint_cb.best_model_path,
            output_path / f"checkpoints_{fold_id}" / "best_model.ckpt",
        )

        del model, dm, trainer
        gc.collect()
        torch.cuda.empty_cache()


def predict_valid(cfg: DictConfig, output_path: Path) -> None:
    torch_dtype = torch.float64 if "64" in cfg.exp.precision else torch.float32
    checkpoint_path = (
        output_path / "checkpoints" / "best_model.ckpt"
        if cfg.exp.pred_checkpoint_path is None
        else cfg.exp.pred_checkpoint_path
    )
    model_module = LeapLightningModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    if cfg.exp.ema.use_ema:
        model_module.model = model_module.model_ema.module
    model = model_module.model

    dm = LeapLightningDataModule(cfg)
    dataloader = dm.val2_dataloader()
    preds1 = []
    preds2 = []
    preds3 = []
    labels1 = []
    labels2 = []
    labels3 = []
    model = model.to("cuda")
    model.eval()
    for (
        x,
        x_cat,
        x1,
        x1_cat,
        x2,
        x2_cat,
        x3,
        x3_cat,
        _,
        y_class1,
        y_class2,
        y_class3,
        _,
        _,
        _,
    ) in tqdm(dataloader):
        x, x_cat, x1, x1_cat, x2, x2_cat, x3, x3_cat = (
            x.to("cuda"),
            x_cat.to("cuda"),
            x1.to("cuda"),
            x1_cat.to("cuda"),
            x2.to("cuda"),
            x2_cat.to("cuda"),
            x3.to("cuda"),
            x3_cat.to("cuda"),
        )
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x_cat = torch.flatten(x_cat, start_dim=0, end_dim=1)
        x1 = torch.flatten(x1, start_dim=0, end_dim=1)
        x1_cat = torch.flatten(x1_cat, start_dim=0, end_dim=1)
        x2 = torch.flatten(x2, start_dim=0, end_dim=1)
        x2_cat = torch.flatten(x2_cat, start_dim=0, end_dim=1)
        x3 = torch.flatten(x3, start_dim=0, end_dim=1)
        x3_cat = torch.flatten(x3_cat, start_dim=0, end_dim=1)

        with torch.no_grad():
            out1 = model(
                x.to(torch_dtype),
                x_cat.to(torch.long),
                x1.to(torch_dtype),
                x1_cat.to(torch.long),
            ).softmax(dim=1)
            out2 = model(
                x.to(torch_dtype),
                x_cat.to(torch.long),
                x2.to(torch_dtype),
                x2_cat.to(torch.long),
            ).softmax(dim=1)
            out3 = model(
                x.to(torch_dtype),
                x_cat.to(torch.long),
                x3.to(torch_dtype),
                x3_cat.to(torch.long),
            ).softmax(dim=1)

        preds1.append(out1.cpu().to(torch.float64))
        preds2.append(out2.cpu().to(torch.float64))
        preds3.append(out3.cpu().to(torch.float64))
        labels1.append(y_class1.flatten().cpu())
        labels2.append(y_class2.flatten().cpu())
        labels3.append(y_class3.flatten().cpu())
        if cfg.debug:
            break

    with utils.trace("save predict"):
        preds1 = np.concatenate(preds1, axis=0)
        preds2 = np.concatenate(preds2, axis=0)
        preds3 = np.concatenate(preds3, axis=0)
        labels1 = np.concatenate(labels1)
        labels2 = np.concatenate(labels2)
        labels3 = np.concatenate(labels3)

        predict1_df = pd.DataFrame(
            preds1, columns=[i for i in range(preds1.shape[1])]
        ).reset_index()
        predict1_df.to_parquet(output_path / "val2_predict1.parquet")
        print(predict1_df)

        predict2_df = pd.DataFrame(
            preds2, columns=[i for i in range(preds2.shape[1])]
        ).reset_index()
        predict2_df.to_parquet(output_path / "val2_predict2.parquet")
        print(predict2_df)

        predict3_df = pd.DataFrame(
            preds3, columns=[i for i in range(preds3.shape[1])]
        ).reset_index()
        predict3_df.to_parquet(output_path / "val2_predict3.parquet")
        print(predict3_df)

        original_label1_df = pd.DataFrame(labels1, columns=[0]).reset_index()
        original_label1_df.to_parquet(output_path / "val2_label1.parquet")
        print(original_label1_df)
        original_label2_df = pd.DataFrame(labels2, columns=[0]).reset_index()
        original_label2_df.to_parquet(output_path / "val2_label2.parquet")
        print(original_label1_df)
        original_label3_df = pd.DataFrame(labels3, columns=[0]).reset_index()
        original_label3_df.to_parquet(output_path / "val2_label3.parquet")
        print(original_label3_df)

        valid_preds = preds1[:, 0]  # classの番号
        valid_y_class = labels1 == 0

        thresholds = [0.5, 0.7, 0.9, 0.99, 0.999, 0.9999, 0.99999]
        for threshold in thresholds:
            y_pred = (valid_preds > threshold).astype(np.int64)
            # リコールと精度を計算
            num_cand = np.sum(y_pred)
            precision = precision_score(valid_y_class, y_pred)
            recall = recall_score(valid_y_class, y_pred)
            print(
                f"Threshold: {threshold:.5f}, Num:{num_cand}, Precision: {precision:.5f}, Recall: {recall:.5f}"
            )


def predict_test(cfg: DictConfig, output_path: Path) -> None:
    torch_dtype = torch.float64 if "64" in cfg.exp.precision else torch.float32
    checkpoint_path = (
        output_path / "checkpoints" / "best_model.ckpt"
        if cfg.exp.pred_checkpoint_path is None
        else cfg.exp.pred_checkpoint_path
    )
    model_module = LeapLightningModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    if cfg.exp.ema.use_ema:
        model_module.model = model_module.model_ema.module
    model = model_module.model

    dm = LeapLightningDataModule(cfg)
    dataloader = dm.test_dataloader()
    preds1 = []
    preds2 = []
    preds3 = []
    model = model.to("cuda")
    model.eval()
    for x, x_cat, x1, x1_cat, x2, x2_cat, x3, x3_cat, __loader__ in tqdm(dataloader):
        x = x.to("cuda").to(torch_dtype)
        x_cat = x_cat.to("cuda").to(torch.long)
        x1 = x1.to("cuda").to(torch_dtype)
        x1_cat = x1_cat.to("cuda").to(torch.long)
        x2 = x2.to("cuda").to(torch_dtype)
        x2_cat = x2_cat.to("cuda").to(torch.long)
        x3 = x3.to("cuda").to(torch_dtype)
        x3_cat = x3_cat.to("cuda").to(torch.long)

        x = torch.flatten(x, start_dim=0, end_dim=1)
        x_cat = torch.flatten(x_cat, start_dim=0, end_dim=1)
        x1 = torch.flatten(x1, start_dim=0, end_dim=1)
        x1_cat = torch.flatten(x1_cat, start_dim=0, end_dim=1)
        x2 = torch.flatten(x2, start_dim=0, end_dim=1)
        x2_cat = torch.flatten(x2_cat, start_dim=0, end_dim=1)
        x3 = torch.flatten(x3, start_dim=0, end_dim=1)
        x3_cat = torch.flatten(x3_cat, start_dim=0, end_dim=1)
        with torch.no_grad():
            out1 = model(x, x_cat, x1, x1_cat).softmax(dim=1)
            out2 = model(x, x_cat, x2, x2_cat).softmax(dim=1)
            out3 = model(x, x_cat, x3, x3_cat).softmax(dim=1)

        preds1.append(out1.detach().cpu().numpy())
        preds2.append(out2.detach().cpu().numpy())
        preds3.append(out3.detach().cpu().numpy())

    preds = np.concatenate(preds1, axis=0)
    original_predict_df = pd.DataFrame(
        preds, columns=[i for i in range(preds.shape[1])]
    ).reset_index()
    original_predict_df.to_parquet(output_path / "test_predict1.parquet")
    print(original_predict_df)

    preds = np.concatenate(preds2, axis=0)
    original_predict_df = pd.DataFrame(
        preds, columns=[i for i in range(preds.shape[1])]
    ).reset_index()
    original_predict_df.to_parquet(output_path / "test_predict2.parquet")
    print(original_predict_df)

    preds = np.concatenate(preds3, axis=0)
    original_predict_df = pd.DataFrame(
        preds, columns=[i for i in range(preds.shape[1])]
    ).reset_index()
    original_predict_df.to_parquet(output_path / "test_predict3.parquet")
    print(original_predict_df)


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
        project="kaggle-leap-binary",
        mode="disabled" if cfg.debug else None,
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
