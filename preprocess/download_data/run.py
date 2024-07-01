import glob
import json
import os
import shutil
import sys
import time
from glob import glob
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import webdataset as wds
import xarray as xr
from google.cloud import storage
from huggingface_hub import snapshot_download
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

MLBackendType = Literal["tensorflow", "pytorch"]
TMP_DIR = Path("/kaggle/working/tmp")


class DataUtils:
    def __init__(
        self,
        grid_info,
        input_mean,
        input_max,
        input_min,
        output_scale,
        ml_backend: MLBackendType = "pytorch",
    ):
        self.data_path = None
        self.input_vars = []
        self.target_vars = []
        self.input_feature_len = None
        self.target_feature_len = None
        self.grid_info = grid_info
        self.level_name = "lev"
        self.sample_name = "sample"
        self.num_levels = len(self.grid_info["lev"])
        self.num_latlon = len(
            self.grid_info["ncol"]
        )  # number of unique lat/lon grid points
        # make area-weights
        self.grid_info["area_wgt"] = self.grid_info["area"] / self.grid_info[
            "area"
        ].mean(dim="ncol")
        self.area_wgt = self.grid_info["area_wgt"].values
        # map ncol to nsamples dimension
        # to_xarray = {'area_wgt':(self.sample_name,np.tile(self.grid_info['area_wgt'], int(n_samples/len(self.grid_info['ncol']))))}
        # to_xarray = xr.Dataset(to_xarray)
        self.input_mean = input_mean
        self.input_max = input_max
        self.input_min = input_min
        self.output_scale = output_scale
        self.normalize = True
        self.lats, self.lats_indices = np.unique(
            self.grid_info["lat"].values, return_index=True
        )
        self.lons, self.lons_indices = np.unique(
            self.grid_info["lon"].values, return_index=True
        )
        self.sort_lat_key = np.argsort(
            self.grid_info["lat"].values[np.sort(self.lats_indices)]
        )
        self.sort_lon_key = np.argsort(
            self.grid_info["lon"].values[np.sort(self.lons_indices)]
        )
        self.indextolatlon = {
            i: (
                self.grid_info["lat"].values[i % self.num_latlon],
                self.grid_info["lon"].values[i % self.num_latlon],
            )
            for i in range(self.num_latlon)
        }

        self.ml_backend = ml_backend
        self.tf = None
        self.torch = None

        if self.ml_backend == "tensorflow":
            self.successful_backend_import = False

            try:
                import tensorflow as tf

                self.tf = tf
                self.successful_backend_import = True
            except ImportError:
                raise ImportError("Tensorflow is not installed.")

        elif self.ml_backend == "pytorch":
            self.successful_backend_import = False

            try:
                import torch

                self.torch = torch
                self.successful_backend_import = True
            except ImportError:
                raise ImportError("PyTorch is not installed.")

        def find_keys(dictionary, value):
            keys = []
            for key, val in dictionary.items():
                if val[0] == value:
                    keys.append(key)
            return keys

        indices_list = []
        for lat in self.lats:
            indices = find_keys(self.indextolatlon, lat)
            indices_list.append(indices)
        indices_list.sort(key=lambda x: x[0])
        self.lat_indices_list = indices_list

        self.hyam = self.grid_info["hyam"].values
        self.hybm = self.grid_info["hybm"].values
        self.p0 = 1e5  # code assumes this will always be a scalar
        self.ps_index = None

        self.pressure_grid_train = None
        self.pressure_grid_val = None
        self.pressure_grid_scoring = None
        self.pressure_grid_test = None

        self.dp_train = None
        self.dp_val = None
        self.dp_scoring = None
        self.dp_test = None

        self.train_regexps = None
        self.train_stride_sample = None
        self.train_filelist = None
        self.val_regexps = None
        self.val_stride_sample = None
        self.val_filelist = None
        self.scoring_regexps = None
        self.scoring_stride_sample = None
        self.scoring_filelist = None
        self.test_regexps = None
        self.test_stride_sample = None
        self.test_filelist = None

        self.full_vars = False

        # physical constants from E3SM_ROOT/share/util/shr_const_mod.F90
        self.grav = 9.80616  # acceleration of gravity ~ m/s^2
        self.cp = 1.00464e3  # specific heat of dry air   ~ J/kg/K
        self.lv = 2.501e6  # latent heat of evaporation ~ J/kg
        self.lf = 3.337e5  # latent heat of fusion      ~ J/kg
        self.lsub = self.lv + self.lf  # latent heat of sublimation ~ J/kg
        self.rho_air = (
            101325 / (6.02214e26 * 1.38065e-23 / 28.966) / 273.15
        )  # density of dry air at STP  ~ kg/m^3
        # ~ 1.2923182846924677
        # SHR_CONST_PSTD/(SHR_CONST_RDAIR*SHR_CONST_TKFRZ)
        # SHR_CONST_RDAIR   = SHR_CONST_RGAS/SHR_CONST_MWDAIR
        # SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ
        self.rho_h20 = 1.0e3  # density of fresh water     ~ kg/m^ 3

        self.v1_inputs = [
            "state_t",
            "state_q0001",
            "state_ps",
            "pbuf_SOLIN",
            "pbuf_LHFLX",
            "pbuf_SHFLX",
        ]

        self.v1_outputs = [
            "ptend_t",
            "ptend_q0001",
            "cam_out_NETSW",
            "cam_out_FLWDS",
            "cam_out_PRECSC",
            "cam_out_PRECC",
            "cam_out_SOLS",
            "cam_out_SOLL",
            "cam_out_SOLSD",
            "cam_out_SOLLD",
        ]

        self.v2_inputs = [
            "state_t",
            "state_q0001",
            "state_q0002",
            "state_q0003",
            "state_u",
            "state_v",
            "state_ps",
            "pbuf_SOLIN",
            "pbuf_LHFLX",
            "pbuf_SHFLX",
            "pbuf_TAUX",
            "pbuf_TAUY",
            "pbuf_COSZRS",
            "cam_in_ALDIF",
            "cam_in_ALDIR",
            "cam_in_ASDIF",
            "cam_in_ASDIR",
            "cam_in_LWUP",
            "cam_in_ICEFRAC",
            "cam_in_LANDFRAC",
            "cam_in_OCNFRAC",
            "cam_in_SNOWHICE",
            "cam_in_SNOWHLAND",
            "pbuf_ozone",  # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3
            "pbuf_CH4",
            "pbuf_N2O",
        ]

        self.v2_outputs = [
            "ptend_t",
            "ptend_q0001",
            "ptend_q0002",
            "ptend_q0003",
            "ptend_u",
            "ptend_v",
            "cam_out_NETSW",
            "cam_out_FLWDS",
            "cam_out_PRECSC",
            "cam_out_PRECC",
            "cam_out_SOLS",
            "cam_out_SOLL",
            "cam_out_SOLSD",
            "cam_out_SOLLD",
        ]

        self.var_lens = {  # inputs
            "state_t": self.num_levels,
            "state_q0001": self.num_levels,
            "state_q0002": self.num_levels,
            "state_q0003": self.num_levels,
            "state_u": self.num_levels,
            "state_v": self.num_levels,
            "state_ps": 1,
            "pbuf_SOLIN": 1,
            "pbuf_LHFLX": 1,
            "pbuf_SHFLX": 1,
            "pbuf_TAUX": 1,
            "pbuf_TAUY": 1,
            "pbuf_COSZRS": 1,
            "cam_in_ALDIF": 1,
            "cam_in_ALDIR": 1,
            "cam_in_ASDIF": 1,
            "cam_in_ASDIR": 1,
            "cam_in_LWUP": 1,
            "cam_in_ICEFRAC": 1,
            "cam_in_LANDFRAC": 1,
            "cam_in_OCNFRAC": 1,
            "cam_in_SNOWHICE": 1,
            "cam_in_SNOWHLAND": 1,
            "pbuf_ozone": self.num_levels,
            "pbuf_CH4": self.num_levels,
            "pbuf_N2O": self.num_levels,
            # outputs
            "ptend_t": self.num_levels,
            "ptend_q0001": self.num_levels,
            "ptend_q0002": self.num_levels,
            "ptend_q0003": self.num_levels,
            "ptend_u": self.num_levels,
            "ptend_v": self.num_levels,
            "cam_out_NETSW": 1,
            "cam_out_FLWDS": 1,
            "cam_out_PRECSC": 1,
            "cam_out_PRECC": 1,
            "cam_out_SOLS": 1,
            "cam_out_SOLL": 1,
            "cam_out_SOLSD": 1,
            "cam_out_SOLLD": 1,
        }

        self.var_short_names = {
            "ptend_t": "$dT/dt$",
            "ptend_q0001": "$dq/dt$",
            "cam_out_NETSW": "NETSW",
            "cam_out_FLWDS": "FLWDS",
            "cam_out_PRECSC": "PRECSC",
            "cam_out_PRECC": "PRECC",
            "cam_out_SOLS": "SOLS",
            "cam_out_SOLL": "SOLL",
            "cam_out_SOLSD": "SOLSD",
            "cam_out_SOLLD": "SOLLD",
        }

        self.target_energy_conv = {
            "ptend_t": self.cp,
            "ptend_q0001": self.lv,
            "ptend_q0002": self.lv,
            "ptend_q0003": self.lv,
            "ptend_wind": None,
            "cam_out_NETSW": 1.0,
            "cam_out_FLWDS": 1.0,
            "cam_out_PRECSC": self.lv * self.rho_h20,
            "cam_out_PRECC": self.lv * self.rho_h20,
            "cam_out_SOLS": 1.0,
            "cam_out_SOLL": 1.0,
            "cam_out_SOLSD": 1.0,
            "cam_out_SOLLD": 1.0,
        }

        # for metrics

        self.input_train = None
        self.target_train = None
        self.preds_train = None
        self.samplepreds_train = None
        self.target_weighted_train = {}
        self.preds_weighted_train = {}
        self.samplepreds_weighted_train = {}
        self.metrics_train = []
        self.metrics_idx_train = {}
        self.metrics_var_train = {}

        self.input_val = None
        self.target_val = None
        self.preds_val = None
        self.samplepreds_val = None
        self.target_weighted_val = {}
        self.preds_weighted_val = {}
        self.samplepreds_weighted_val = {}
        self.metrics_val = []
        self.metrics_idx_val = {}
        self.metrics_var_val = {}

        self.input_scoring = None
        self.target_scoring = None
        self.preds_scoring = None
        self.samplepreds_scoring = None
        self.target_weighted_scoring = {}
        self.preds_weighted_scoring = {}
        self.samplepreds_weighted_scoring = {}
        self.metrics_scoring = []
        self.metrics_idx_scoring = {}
        self.metrics_var_scoring = {}

        self.input_test = None
        self.target_test = None
        self.preds_test = None
        self.samplepreds_test = None
        self.target_weighted_test = {}
        self.preds_weighted_test = {}
        self.samplepreds_weighted_test = {}
        self.metrics_test = []
        self.metrics_idx_test = {}
        self.metrics_var_test = {}

        self.model_names = []
        self.metrics_names = []
        self.num_CRPS = 32
        self.linecolors = ["#0072B2", "#E69F00", "#882255", "#009E73", "#D55E00"]

    def set_to_v2_vars(self):
        """
        This function sets the inputs and outputs to the V2 subset.
        It also indicates the index of the surface pressure variable.
        """
        self.input_vars = self.v2_inputs
        self.target_vars = self.v2_outputs
        self.ps_index = 360
        self.input_feature_len = 557
        self.target_feature_len = 368
        self.full_vars = True

    def get_xrdata(self, file, file_vars=None):
        """
        This function reads in a file and returns an xarray dataset with the variables specified.
        file_vars must be a list of strings.
        """
        ds = xr.open_dataset(file, engine="netcdf4")
        if file_vars is not None:
            ds = ds[file_vars]
        ds = ds.merge(self.grid_info[["lat", "lon"]])
        ds = ds.where((ds["lat"] > -999) * (ds["lat"] < 999), drop=True)
        ds = ds.where((ds["lon"] > -999) * (ds["lon"] < 999), drop=True)
        return ds

    def get_input(self, input_file):
        """
        This function reads in a file and returns an xarray dataset with the input variables for the emulator.
        """
        # read inputs
        return self.get_xrdata(input_file, self.input_vars)

    def get_target(self, ds_input, input_file):
        """
        This function reads in a file and returns an xarray dataset with the target variables for the emulator.
        """
        # read inputs
        ds_target = self.get_xrdata(input_file.replace(".mli.", ".mlo."))
        # each timestep is 20 minutes which corresponds to 1200 seconds
        ds_target["ptend_t"] = (
            ds_target["state_t"] - ds_input["state_t"]
        ) / 1200  # T tendency [K/s]
        ds_target["ptend_q0001"] = (
            ds_target["state_q0001"] - ds_input["state_q0001"]
        ) / 1200  # Q tendency [kg/kg/s]
        if self.full_vars:
            ds_target["ptend_q0002"] = (
                ds_target["state_q0002"] - ds_input["state_q0002"]
            ) / 1200  # Q tendency [kg/kg/s]
            ds_target["ptend_q0003"] = (
                ds_target["state_q0003"] - ds_input["state_q0003"]
            ) / 1200  # Q tendency [kg/kg/s]
            ds_target["ptend_u"] = (
                ds_target["state_u"] - ds_input["state_u"]
            ) / 1200  # U tendency [m/s/s]
            ds_target["ptend_v"] = (
                ds_target["state_v"] - ds_input["state_v"]
            ) / 1200  # V tendency [m/s/s]
        ds_target = ds_target[self.target_vars]
        return ds_target

    def gen(self, file):
        # read inputs
        ds_input = self.get_input(file)
        # read targets
        ds_target = self.get_target(ds_input, file)

        ds_input = ds_input.drop_vars(["lat", "lon"])

        # stack
        # ds = ds.stack({'batch':{'sample','ncol'}})
        ds_input = ds_input.stack({"batch": {"ncol"}})
        ds_input = ds_input.to_stacked_array("mlvar", sample_dims=["batch"], name="mli")
        # dso = dso.stack({'batch':{'sample','ncol'}})
        ds_target = ds_target.stack({"batch": {"ncol"}})
        ds_target = ds_target.to_stacked_array(
            "mlvar", sample_dims=["batch"], name="mlo"
        )
        return (ds_input.values, ds_target.values)


def upload_directory_to_gcs(local_directory, bucket_name, gcs_directory):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    files_to_upload = []
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            gcs_file_path = os.path.join(
                gcs_directory, os.path.relpath(local_file_path, local_directory)
            )
            files_to_upload.append((local_file_path, gcs_file_path))

    progress_bar = tqdm(total=len(files_to_upload), unit="file")

    for local_file_path, gcs_file_path in files_to_upload:
        blob = bucket.blob(gcs_file_path)
        blob.upload_from_filename(local_file_path, if_generation_match=None)
        progress_bar.update(1)
        progress_bar.set_postfix(file=os.path.basename(local_file_path))

    progress_bar.close()
    print(f"All files uploaded successfully to {gcs_directory}.")


def dowload_month(cfg: DictConfig, month_dir: str, data_utils: DataUtils) -> None:
    # clear tmp dir
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(exist_ok=True)

    month_dir = Path(month_dir)
    # Download
    print("Downloading...")
    allow_patterns = (
        str(month_dir / f"*{month_dir.name}-01-*.nc")  # 1日だけ
        if cfg.debug
        else str(month_dir / f"*{month_dir.name}-*-*.nc")
    )
    print(f"{allow_patterns=}")

    retry_count = 0
    while True:
        try:
            _ = snapshot_download(
                repo_id=cfg.exp.repo_id,
                allow_patterns=allow_patterns,
                local_dir=TMP_DIR,
                cache_dir=TMP_DIR / "cache",
                # local_dir_use_symlinks=False,  # キャッシュしない
                repo_type="dataset",
                resume_download=True,
            )
            break
        except Exception as e:
            print(f"ConnectionError: {e}. Retrying in 1s...")
            if retry_count == 3:
                raise e
            retry_count += 1
            time.sleep(retry_count)

    save_dir = Path(cfg.exp.save_dir) / str(month_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"{save_dir=}")

    input_paths = glob(str(TMP_DIR / month_dir / "*.mli.*.nc"))

    for i, input_path in enumerate(tqdm(input_paths)):
        input_data, output_data = data_utils.gen(input_path)
        input_data = np.delete(input_data, 375, 1)

        array = np.concatenate([input_data, output_data], axis=1)

        input_path = Path(input_path)
        file_name_text = input_path.stem.split(".")[-1]

        # numpyとして保存
        np.save(save_dir / f"{file_name_text}.npy", array)

        if i == 0:
            print(f"{array.shape=}")

        if cfg.debug:
            break


def make_webdataset(cfg: DictConfig, exp_name) -> None:
    grid_path = "/kaggle/working/misc/grid_info/ClimSim_low-res_grid-info.nc"
    norm_path = "/kaggle/working/misc/preprocessing/normalizations/"
    grid_info = xr.open_dataset(grid_path)
    input_mean = xr.open_dataset(norm_path + "inputs/input_mean.nc")
    input_max = xr.open_dataset(norm_path + "inputs/input_max.nc")
    input_min = xr.open_dataset(norm_path + "inputs/input_min.nc")
    output_scale = xr.open_dataset(norm_path + "outputs/output_scale.nc")

    data_utils = DataUtils(
        grid_info=grid_info,
        input_mean=input_mean,
        input_max=input_max,
        input_min=input_min,
        output_scale=output_scale,
    )
    data_utils.set_to_v2_vars()
    data_utils.normalize = False

    """
    month_dirs = (
        [f"train/0001-{str(m).zfill(2)}" for m in range(2, 13)]
        + [f"train/000{y}-{str(m).zfill(2)}" for y in range(2, 9) for m in range(1, 13)]
        + ["train/0009-01"]
    )
    """
    month_dirs = (
        [f"train/0005-{str(m).zfill(2)}" for m in range(2, 13)]
        + [f"train/000{y}-{str(m).zfill(2)}" for y in range(6, 7) for m in range(1, 13)]
        + ["train/0007-01"]
    )
    if cfg.exp.break_n_months:
        month_dirs = month_dirs[: cfg.exp.break_n_months]
    if cfg.debug:
        month_dirs = month_dirs[:1]

    for month_dir in month_dirs:
        print(f"Processing {month_dir}")
        dowload_month(cfg, month_dir, data_utils)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.preprocess_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    make_webdataset(cfg, exp_name)


if __name__ == "__main__":
    main()
