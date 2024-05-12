"""
y のoutlier を除外してから各統計量を計算する
"""

import pickle
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr
from tqdm import tqdm

debug = False
n_sampling = 9000 if debug else int(625000 * 0.8)
iter_sampling = 10 if debug else 100

output_path = Path("normalize_v6_feat")
output_path.mkdir(exist_ok=True)


# physical constatns from (E3SM_ROOT/share/util/shr_const_mod.F90)
grav = 9.80616  # acceleration of gravity ~ m/s^2
cp = 1.00464e3  # specific heat of dry air   ~ J/kg/K
lv = 2.501e6  # latent heat of evaporation ~ J/kg
lf = 3.337e5  # latent heat of fusion      ~ J/kg
ls = lv + lf  # latent heat of sublimation ~ J/kg
rho_air = 101325.0 / (6.02214e26 * 1.38065e-23 / 28.966) / 273.15
rho_h20 = 1.0e3  # density of fresh water     ~ kg/m^ 3


def cal_y_min_max(df):
    print("cal_y_min_max")
    y_sample_min_list = []
    y_sample_max_list = []
    for i in tqdm(range(iter_sampling)):
        sample_df = df[:, 557:].sample(n_sampling, seed=i)
        y_sample_min = sample_df.min().to_numpy().ravel()
        y_sample_min_list.append(y_sample_min)
        y_sample_max = sample_df.max().to_numpy().ravel()
        y_sample_max_list.append(y_sample_max)

    y_sample_min_max = np.max(y_sample_min_list, axis=0)
    np.save(output_path / "y_sample_min_max.npy", y_sample_min)
    y_sample_max_min = np.min(y_sample_max_list, axis=0)
    np.save(output_path / "y_sample_max_min.npy", y_sample_max)
    return y_sample_min_max, y_sample_max_min


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
    # len=60
    water_array = x_array[:, 60:120] + x_array[:, 120:180] + x_array[:, 180:240]
    water_energy_array = water_array * pressures_array * lv / grav
    temp_energy_array = x_array[:, 0:60] * pressures_array * cp / grav
    temp_water_energy_array = water_energy_array + temp_energy_array
    u_energy_array = x_array[:, 240:300] * pressures_array / grav
    v_energy_array = x_array[:, 300:360] * pressures_array / grav
    wind_energy_array = x_array[:, 240:300] ** 2 + x_array[:, 300:360] ** 2
    wind_vec_array = np.sqrt(wind_energy_array)

    # scalar
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
        "pressures": pressures_array,
        "water": water_array,
        "water_energy": water_energy_array,
        "temp_energy": temp_energy_array,
        "temp_water_energy": temp_water_energy_array,
        "u_energy": u_energy_array,
        "v_energy": v_energy_array,
        "wind_energy": wind_energy_array,
        "wind_vec": wind_vec_array,
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


def cal_stats_x_y(df, y_sample_min_max, y_sample_max_min, diff_rate, features_dict):
    print(f"cal_stats_x_y: {diff_rate=}")

    y_diff = (y_sample_max_min - y_sample_min_max) * diff_rate
    y_lower_bound = y_sample_min_max - y_diff
    np.save(output_path / f"y_lower_bound_{diff_rate}.npy", y_lower_bound)

    y_upper_bound = y_sample_max_min + y_diff
    np.save(output_path / f"y_upper_bound_{diff_rate}.npy", y_upper_bound)

    y = df[:, 557:].to_numpy()
    y[y < y_lower_bound] = np.nan
    y[y > y_upper_bound] = np.nan
    y_nanmean = np.nanmean(y, axis=0)
    np.save(output_path / f"y_nanmean_{diff_rate}.npy", y_nanmean)
    y_nanmin = np.nanmin(y, axis=0)
    np.save(output_path / f"y_nanmin_{diff_rate}.npy", y_nanmin)
    y_nanmax = np.nanmax(y, axis=0)
    np.save(output_path / f"y_nanmax_{diff_rate}.npy", y_nanmax)
    y_nanstd = np.nanstd(y, axis=0)
    np.save(output_path / f"y_nanstd_{diff_rate}.npy", y_nanstd)

    y_rms_np = np.sqrt(np.nanmean(y * y, axis=0)).ravel()
    np.save(output_path / f"y_rms_{diff_rate}.npy", y_rms_np)

    y_sub = y - y_nanmean
    y_rms_sub_np = np.sqrt(np.nanmean(y_sub * y_sub, axis=0)).ravel()
    np.save(output_path / f"y_rms_sub_{diff_rate}.npy", y_rms_sub_np)
    print(
        f"{y_nanmean.shape=}, {y_nanmin.shape=}, {y_nanmax.shape=}, {y_nanstd.shape=}, {y_rms_np.shape=}, {y_rms_sub_np.shape=}"
    )

    print("start x")
    ### x
    # y で outlier とならない範囲のデータを抽出
    eps = 1e-60
    filter_bool = np.all(
        df[:, 557:].to_numpy() >= y_lower_bound - eps, axis=1
    ) & np.all(df[:, 557:].to_numpy() <= y_upper_bound + eps, axis=1)
    print(f"{df.shape=}")
    filter_df = df.filter(filter_bool)
    print(f"{df.shape=}, {filter_df.shape=}")

    mean_feat_dict = {}
    std_feat_dict = {}
    for key, values in features_dict.items():
        array = values[filter_bool]
        x_mean = array.mean(axis=0)
        x_std = array.std(axis=0)
        mean_feat_dict[key] = x_mean
        std_feat_dict[key] = x_std

    with open(output_path / f"x_mean_feat_dict_{diff_rate}.pkl", "wb") as f:
        pickle.dump(mean_feat_dict, f)
    with open(output_path / f"x_std_feat_dict_{diff_rate}.pkl", "wb") as f:
        pickle.dump(std_feat_dict, f)


if __name__ == "__main__":
    df = pl.read_parquet("../input/train.parquet", n_rows=10000 if debug else None)

    y_sample_min_max, y_sample_max_min = cal_y_min_max(df)

    pressures = cal_pressures(df)
    features_dict = process_features(df[:, 1:557].to_numpy(), pressures)

    for diff_rate in [16.0, 4.0, 1.0, 0.5, 0.1, 0]:
        cal_stats_x_y(df, y_sample_min_max, y_sample_max_min, diff_rate, features_dict)
