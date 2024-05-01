import os
import sys
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import xarray as xr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

v2_inputs = [
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

v2_outputs = [
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

vertically_resolved = [
    "state_t",
    "state_q0001",
    "state_q0002",
    "state_q0003",
    "state_u",
    "state_v",
    "pbuf_ozone",
    "pbuf_CH4",
    "pbuf_N2O",
    "ptend_t",
    "ptend_q0001",
    "ptend_q0002",
    "ptend_q0003",
    "ptend_u",
    "ptend_v",
]
ablated_vars = ["ptend_q0001", "ptend_q0002", "ptend_q0003", "ptend_u", "ptend_v"]

v2_vars = v2_inputs + v2_outputs
train_col_names = []
ablated_col_names = []
for var in v2_vars:
    if var in vertically_resolved:
        for i in range(60):
            train_col_names.append(var + "_" + str(i))
            if i < 12 and var in ablated_vars:
                ablated_col_names.append(var + "_" + str(i))
    else:
        train_col_names.append(var)


def dataset_to_1d_numpy(ds, var_order):
    data_list = []
    for var_name in var_order:
        data = ds[var_name].values.ravel()
        data_list.append(data)

    combined_data = np.concatenate(data_list)
    return combined_data


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.preprocess_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    index_SNOWHICE = train_col_names.index("cam_in_SNOWHICE")
    print(f"{index_SNOWHICE=}")

    norm_path = "/kaggle/working/misc/preprocessing/normalizations/"
    input_mean = xr.open_dataset(norm_path + "inputs/input_mean.nc")
    input_max = xr.open_dataset(norm_path + "inputs/input_max.nc")
    input_min = xr.open_dataset(norm_path + "inputs/input_min.nc")
    output_scale = xr.open_dataset(norm_path + "outputs/output_scale.nc")

    use_cols = [col for col in v2_inputs if col != "cam_in_SNOWHICE"]
    input_mean_np = dataset_to_1d_numpy(input_mean, use_cols)
    input_max_np = dataset_to_1d_numpy(input_max, use_cols)
    input_min_np = dataset_to_1d_numpy(input_min, use_cols)
    output_scale_np = dataset_to_1d_numpy(output_scale, v2_outputs)

    # shape
    print(
        f"{input_mean_np.shape=}",
        f"{input_max_np.shape=}",
        f"{input_min_np.shape=}",
        f"{output_scale_np.shape=}",
    )
    # save
    np.save(output_path / "input_mean.npy", input_mean_np)
    np.save(output_path / "input_max.npy", input_max_np)
    np.save(output_path / "input_min.npy", input_min_np)
    np.save(output_path / "output_scale.npy", output_scale_np)


if __name__ == "__main__":
    main()
