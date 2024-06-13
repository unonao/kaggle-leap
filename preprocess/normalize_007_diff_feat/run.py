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

from utils.humidity import ThermLibNumpy, cal_normalized_lfh, cal_specific2relative_coef

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


def process_features(cfg, x_array, y_array, pressures_array):
    grid_path = "/kaggle/working/misc/grid_info/ClimSim_low-res_grid-info.nc"
    grid_info = xr.open_dataset(grid_path)
    hyam = grid_info["hyam"].to_numpy()
    hybm = grid_info["hybm"].to_numpy()

    specific2relative_coef = cal_specific2relative_coef(
        temperature_array=x_array[:, 0:60],
        near_surface_air_pressure=x_array[:, 360],
        hyam=hyam,
        hybm=hybm,
        method=cfg.exp.rh_method,
    )
    relative_humidity = specific2relative_coef * x_array[:, 60:120]
    # len=60
    y_relative_humidity = specific2relative_coef * y_array[:, 60:120]

    q1_log_array = np.log1p(x_array[:, 60:120] * 1e9)
    q2_log_array = np.log1p(x_array[:, 120:180] * 1e9)
    q3_log_array = np.log1p(x_array[:, 180:240] * 1e9)

    cloud_water_array = x_array[:, 120:180] + x_array[:, 180:240]
    cloud_water_log_array = np.log1p(cloud_water_array * 1e9)
    water_array = x_array[:, 60:120] + x_array[:, 120:180] + x_array[:, 180:240]

    q2q3_mean_array = (x_array[:, 120:180] + x_array[:, 180:240]) / 2
    uv_mean_array = (x_array[:, 240:300] + x_array[:, 300:360]) / 2
    pbuf_mean_array = (
        x_array[:, 376 : 376 + 60]
        + x_array[:, 376 + 60 : 376 + 120]
        + x_array[:, 376 + 120 : 376 + 180]
    ) / 3

    t_diff_array = np.diff(
        x_array[:, 0:60], axis=1, append=0
    )  # 地上に近い方からの温度差を入れる
    q1_diff_array = np.diff(x_array[:, 60:120], axis=1, append=0)
    print(q1_diff_array.shape)
    q2_diff_array = np.diff(x_array[:, 120:180], axis=1, append=0)
    q3_diff_array = np.diff(x_array[:, 180:240], axis=1, append=0)
    u_diff_array = np.diff(x_array[:, 240:300], axis=1, append=0)
    v_diff_array = np.diff(x_array[:, 300:360], axis=1, append=0)
    ozone_diff_array = np.diff(x_array[:, 376 : 376 + 60], axis=1, append=0)
    ch4_diff_array = np.diff(x_array[:, 376 + 60 : 376 + 120], axis=1, append=0)
    n2o_diff_array = np.diff(x_array[:, 376 + 120 : 376 + 180], axis=1, append=0)
    q2q3_mean_array_diff = np.diff(q2q3_mean_array, axis=1, append=0)
    uv_mean_array_diff = np.diff(uv_mean_array, axis=1, append=0)
    pbuf_mean_array_diff = np.diff(pbuf_mean_array, axis=1, append=0)

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
        "y_relative_humidity": y_relative_humidity,
        "base": x_array[:, :556],
        "relative_humidity": relative_humidity,
        "q1_log": q1_log_array,
        "q2_log": q2_log_array,
        "q3_log": q3_log_array,
        "cloud_water": cloud_water_array,
        "cloud_water_log": cloud_water_log_array,
        "pressures": pressures_array,
        "water": water_array,
        "q2q3_mean": q2q3_mean_array,
        "uv_mean": uv_mean_array,
        "pbuf_mean": pbuf_mean_array,
        "t_diff": t_diff_array,
        "q1_diff": q1_diff_array,
        "q2_diff": q2_diff_array,
        "q3_diff": q3_diff_array,
        "u_diff": u_diff_array,
        "v_diff": v_diff_array,
        "ozone_diff": ozone_diff_array,
        "ch4_diff": ch4_diff_array,
        "n2o_diff": n2o_diff_array,
        "q2q3_mean_diff": q2q3_mean_array_diff,
        "uv_mean_diff": uv_mean_array_diff,
        "pbuf_mean_diff": pbuf_mean_array_diff,
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
    mean_feat_dict["u_all"] = np.nanmean(features_dict["base"][:, 240:300])
    mean_feat_dict["v_all"] = np.nanmean(features_dict["base"][:, 300:360])
    mean_feat_dict["ozone_all"] = np.nanmean(features_dict["base"][:, 376 : 376 + 60])
    mean_feat_dict["ch4_all"] = np.nanmean(
        features_dict["base"][:, 376 + 60 : 376 + 120]
    )
    mean_feat_dict["n2o_all"] = np.nanmean(
        features_dict["base"][:, 376 + 120 : 376 + 180]
    )

    std_feat_dict["t_all"] = np.nanstd(features_dict["base"][:, 0:60])
    std_feat_dict["q1_all"] = np.nanstd(features_dict["base"][:, 60:120])
    std_feat_dict["q2_all"] = np.nanstd(features_dict["base"][:, 120:180])
    std_feat_dict["q3_all"] = np.nanstd(features_dict["base"][:, 180:240])
    std_feat_dict["u_all"] = np.nanstd(features_dict["base"][:, 240:300])
    std_feat_dict["v_all"] = np.nanstd(features_dict["base"][:, 300:360])
    std_feat_dict["ozone_all"] = np.nanstd(features_dict["base"][:, 376 : 376 + 60])
    std_feat_dict["ch4_all"] = np.nanstd(features_dict["base"][:, 376 + 60 : 376 + 120])
    std_feat_dict["n2o_all"] = np.nanstd(
        features_dict["base"][:, 376 + 120 : 376 + 180]
    )
    mean_feat_dict["q1_log_all"] = np.nanmean(features_dict["q1_log"])
    mean_feat_dict["q2_log_all"] = np.nanmean(features_dict["q2_log"])
    mean_feat_dict["q3_log_all"] = np.nanmean(features_dict["q3_log"])
    std_feat_dict["q1_log_all"] = np.nanstd(features_dict["q1_log"])
    std_feat_dict["q2_log_all"] = np.nanstd(features_dict["q2_log"])
    std_feat_dict["q3_log_all"] = np.nanstd(features_dict["q3_log"])

    mean_feat_dict["y_relative_humidity_all"] = np.nanmean(
        features_dict["y_relative_humidity"]
    )
    mean_feat_dict["relative_humidity_all"] = np.nanmean(
        features_dict["relative_humidity"]
    )
    mean_feat_dict["cloud_water_all"] = np.nanmean(features_dict["cloud_water"])
    mean_feat_dict["cloud_water_log_all"] = np.nanmean(features_dict["cloud_water_log"])
    mean_feat_dict["pressures_all"] = np.nanmean(features_dict["pressures"])

    std_feat_dict["y_relative_humidity_all"] = np.nanstd(
        features_dict["y_relative_humidity"]
    )
    std_feat_dict["relative_humidity_all"] = np.nanstd(
        features_dict["relative_humidity"]
    )
    std_feat_dict["cloud_water_all"] = np.nanstd(features_dict["cloud_water"])
    std_feat_dict["cloud_water_log_all"] = np.nanstd(features_dict["cloud_water_log"])
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
    features_dict = process_features(
        cfg, df[:, 1:557].to_numpy(), df[:, 557:].to_numpy(), pressures
    )

    cal_stats_x_y(
        df,
        features_dict,
        output_path,
    )


if __name__ == "__main__":
    main()
