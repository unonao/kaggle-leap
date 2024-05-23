import os
import pickle
import sys
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import xarray as xr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# physical constatns from (E3SM_ROOT/share/util/shr_const_mod.F90)
grav = 9.80616  # acceleration of gravity ~ m/s^2
cp = 1.00464e3  # specific heat of dry air   ~ J/kg/K
lv = 2.501e6  # latent heat of evaporation ~ J/kg
lf = 3.337e5  # latent heat of fusion      ~ J/kg
ls = lv + lf  # latent heat of sublimation ~ J/kg
rho_air = 101325.0 / (6.02214e26 * 1.38065e-23 / 28.966) / 273.15
rho_h20 = 1.0e3  # density of fresh water     ~ kg/m^ 3


def cal_pressures(df):
    grid_path = "/kaggle/working/misc/grid_info/ClimSim_low-res_grid-info.nc"
    grid_info = xr.open_dataset(grid_path)
    hyai = grid_info["hyai"].to_numpy()
    hybi = grid_info["hybi"].to_numpy()
    p0 = 1e5
    ps = df["state_ps"].to_numpy()
    pressures_array = hyai * p0 + hybi[None, :] * ps[:, None]
    pressures_array = np.diff(pressures_array, n=1)
    print(f"{pressures_array.shape=}")
    return pressures_array


def process_features(x_array, pressures_array):
    q1_log_array = np.log1p(x_array[:, 60:120] * 1e9)
    q2_log_array = np.log1p(x_array[:, 120:180] * 1e9)
    q3_log_array = np.log1p(x_array[:, 180:240] * 1e9)
    # len=60
    cloud_water_array = x_array[:, 120:180] + x_array[:, 180:240]
    cloud_watter_log_array = np.log1p(cloud_water_array * 1e9)
    water_array = x_array[:, 60:120] + x_array[:, 120:180] + x_array[:, 180:240]
    water_energy_array = water_array * pressures_array * lv / grav
    temp_energy_array = x_array[:, 0:60] * pressures_array * cp / grav
    temp_water_energy_array = water_energy_array + temp_energy_array
    u_energy_array = x_array[:, 240:300] * pressures_array / grav
    v_energy_array = x_array[:, 300:360] * pressures_array / grav
    wind_energy_array = x_array[:, 240:300] ** 2 + x_array[:, 300:360] ** 2
    wind_vec_array = np.sqrt(wind_energy_array)

    # scalar
    q1_sum = x_array[:, 60:120].sum(axis=1)
    q1_sum_log = np.log1p(q1_sum * 1e9)
    q2_sum = x_array[:, 120:180].sum(axis=1)
    q2_sum_log = np.log1p(q2_sum * 1e9)
    q3_sum = x_array[:, 180:240].sum(axis=1)
    q3_sum_log = np.log1p(q3_sum * 1e9)
    sum_energy_array = x_array[:, [361, 362, 363, 371]].sum(axis=1)
    sum_flux_array = x_array[:, [362, 363, 371]].sum(axis=1)
    energy_diff_array = x_array[:, 361] - sum_flux_array
    bowen_ratio_array = x_array[:, 362] / x_array[:, 363]
    sum_surface_stress_array = x_array[:, [364, 365]].sum(axis=1)
    net_radiative_flux_array = x_array[:, 361] * x_array[:, 366] - x_array[:, 371]
    global_solar_irradiance_array = (
        x_array[:, 361] * (1 - x_array[:, 369]) * (1 - x_array[:, 370])
    )
    global_longwave_flux_array = (
        x_array[:, 371] * (1 - x_array[:, 367]) * (1 - x_array[:, 368])
    )

    result_dict = {
        "base": x_array[:, :556],
        "q1_log": q1_log_array,
        "q2_log": q2_log_array,
        "q3_log": q3_log_array,
        "cloud_water": cloud_water_array,
        "cloud_watter_log": cloud_watter_log_array,
        "pressures": pressures_array,
        "water": water_array,
        "water_energy": water_energy_array,
        "temp_energy": temp_energy_array,
        "temp_water_energy": temp_water_energy_array,
        "u_energy": u_energy_array,
        "v_energy": v_energy_array,
        "wind_energy": wind_energy_array,
        "wind_vec": wind_vec_array,
        "q1_sum": q1_sum,
        "q1_sum_log": q1_sum_log,
        "q2_sum": q2_sum,
        "q2_sum_log": q2_sum_log,
        "q3_sum": q3_sum,
        "q3_sum_log": q3_sum_log,
        "sum_energy": sum_energy_array,
        "sum_flux": sum_flux_array,
        "energy_diff": energy_diff_array,
        "bowen_ratio": bowen_ratio_array,
        "sum_surface_stress": sum_surface_stress_array,
        "net_radiative_flux": net_radiative_flux_array,
        "global_solar_irradiance": global_solar_irradiance_array,
        "global_longwave_flux": global_longwave_flux_array,
    }

    return result_dict


def cal_stats_x_y(df, features_dict, output_path):
    print("start x")
    ### x
    print(f"{df.shape=}")
    mean_feat_dict = {}
    std_feat_dict = {}
    for key, array in features_dict.items():
        x_mean = np.nanmean(array, axis=0)
        x_std = np.nanstd(array, axis=0)
        if np.isnan(x_mean).any():
            print(f"{key=}, {x_mean=}")
        mean_feat_dict[key] = x_mean
        std_feat_dict[key] = x_std

    mean_feat_dict["t_all"] = np.nanmean(features_dict["base"][:, 0:60])
    mean_feat_dict["q1_all"] = np.nanmean(features_dict["base"][:, 60:120])
    mean_feat_dict["q2_all"] = np.nanmean(features_dict["base"][:, 120:180])
    mean_feat_dict["q3_all"] = np.nanmean(features_dict["base"][:, 180:240])
    mean_feat_dict["cloud_water_all"] = np.nanmean(features_dict["cloud_water"])
    mean_feat_dict["u_all"] = np.nanmean(features_dict["base"][:, 240:300])
    mean_feat_dict["v_all"] = np.nanmean(features_dict["base"][:, 300:360])
    std_feat_dict["t_all"] = np.nanstd(features_dict["base"][:, 0:60])
    std_feat_dict["q1_all"] = np.nanstd(features_dict["base"][:, 60:120])
    std_feat_dict["q2_all"] = np.nanstd(features_dict["base"][:, 120:180])
    std_feat_dict["q3_all"] = np.nanstd(features_dict["base"][:, 180:240])
    std_feat_dict["cloud_water_all"] = np.nanstd(features_dict["cloud_water"])
    std_feat_dict["u_all"] = np.nanstd(features_dict["base"][:, 240:300])
    std_feat_dict["v_all"] = np.nanstd(features_dict["base"][:, 300:360])
    mean_feat_dict["q1_log_all"] = np.nanmean(features_dict["q1_log"])
    mean_feat_dict["q2_log_all"] = np.nanmean(features_dict["q2_log"])
    mean_feat_dict["q3_log_all"] = np.nanmean(features_dict["q3_log"])
    mean_feat_dict["cloud_water_log_all"] = np.nanmean(
        features_dict["cloud_watter_log"]
    )
    std_feat_dict["q1_log_all"] = np.nanstd(features_dict["q1_log"])
    std_feat_dict["q2_log_all"] = np.nanstd(features_dict["q2_log"])
    std_feat_dict["q3_log_all"] = np.nanstd(features_dict["q3_log"])
    std_feat_dict["cloud_water_log_all"] = np.nanstd(features_dict["cloud_watter_log"])

    mean_feat_dict["pressures_all"] = np.nanmean(features_dict["pressures"])
    std_feat_dict["pressures_all"] = np.nanstd(features_dict["pressures"])

    with open(output_path / "x_mean_feat_dict.pkl", "wb") as f:
        pickle.dump(mean_feat_dict, f)
    with open(output_path / "x_std_feat_dict.pkl", "wb") as f:
        pickle.dump(std_feat_dict, f)

    print("cal_stats_x_y")
    y = df[:, 557:].to_numpy()
    y_nanmean = np.nanmean(y, axis=0)
    np.save(output_path / "y_nanmean.npy", y_nanmean)
    y_nanmin = np.nanmin(y, axis=0)
    np.save(output_path / "y_nanmin.npy", y_nanmin)
    y_nanmax = np.nanmax(y, axis=0)
    np.save(output_path / "y_nanmax.npy", y_nanmax)
    y_nanstd = np.nanstd(y, axis=0)
    np.save(output_path / "y_nanstd.npy", y_nanstd)
    y_rms_np = np.sqrt(np.nanmean(y * y, axis=0)).ravel()
    np.save(output_path / "y_rms.npy", y_rms_np)

    y_sub = y - y_nanmean
    y_rms_sub_np = np.sqrt(np.nanmean(y_sub * y_sub, axis=0)).ravel()
    np.save(output_path / "y_rms_sub.npy", y_rms_sub_np)
    print(
        f"{y_nanmean.shape=}, {y_nanmin.shape=}, {y_nanmax.shape=}, {y_nanstd.shape=}, {y_rms_np.shape=}, {y_rms_sub_np.shape=}"
    )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.preprocess_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    df = pl.read_parquet("input/train.parquet", n_rows=50000 if cfg.debug else None)
    print(df.shape)

    pressures = cal_pressures(df)
    features_dict = process_features(df[:, 1:557].to_numpy(), pressures)

    cal_stats_x_y(
        df,
        features_dict,
        output_path,
    )


if __name__ == "__main__":
    main()
