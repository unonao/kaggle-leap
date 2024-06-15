import gc
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
from scipy.ndimage import uniform_filter1d

from utils.humidity import cal_specific2relative_coef

# physical constatns from (E3SM_ROOT/share/util/shr_const_mod.F90)
grav = 9.80616  # acceleration of gravity ~ m/s^2
cp = 1.00464e3  # specific heat of dry air   ~ J/kg/K
lv = 2.501e6  # latent heat of evaporation ~ J/kg
lf = 3.337e5  # latent heat of fusion      ~ J/kg
ls = lv + lf  # latent heat of sublimation ~ J/kg
rho_air = 101325.0 / (6.02214e26 * 1.38065e-23 / 28.966) / 273.15
rho_h20 = 1.0e3  # density of fresh water     ~ kg/m^ 3


def rolling_mean_std(data, window_size):
    # Rolling mean
    mean = uniform_filter1d(data, size=window_size, axis=1, mode="nearest")

    # Rolling std
    mean_sq = uniform_filter1d(data**2, size=window_size, axis=1, mode="nearest")
    variance = mean_sq - mean**2

    # Replace negative variance with zero
    variance[variance < 0] = 0
    std = np.sqrt(variance)

    return (
        np.nanmean(mean, axis=0),
        np.nanstd(mean, axis=0),
        np.nanmean(std, axis=0),
        np.nanstd(std, axis=0),
    )


def calc_feat(cfg, x_array, y_array, output_path):
    mean_feat_dict = {}
    std_feat_dict = {}

    mean_feat_dict["base"] = np.nanmean(x_array[:, 0:556], axis=0)
    std_feat_dict["base"] = np.nanstd(x_array[:, 0:556], axis=0)

    # y_relative_humidity_all
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
    y_relative_humidity = specific2relative_coef * y_array[:, 60:120]

    mean_feat_dict["y_relative_humidity_all"] = np.nanmean(y_relative_humidity, axis=0)
    std_feat_dict["y_relative_humidity_all"] = np.nanstd(y_relative_humidity, axis=0)
    mean_feat_dict["relative_humidity_all"] = np.nanmean(relative_humidity, axis=0)
    std_feat_dict["relative_humidity_all"] = np.nanstd(relative_humidity, axis=0)
    mean_feat_dict["y_relative_humidity_all"] = np.nanmean(y_relative_humidity)
    mean_feat_dict["relative_humidity_all"] = np.nanmean(relative_humidity)
    std_feat_dict["y_relative_humidity_all"] = np.nanstd(y_relative_humidity)
    std_feat_dict["relative_humidity_all"] = np.nanstd(relative_humidity)

    del specific2relative_coef, relative_humidity, y_relative_humidity
    gc.collect()

    # q1_log_all, q2_log_all, q3_log_all
    q1_log_array = np.log1p(x_array[:, 60:120] * 1e9)
    q2_log_array = np.log1p(x_array[:, 120:180] * 1e9)
    q3_log_array = np.log1p(x_array[:, 180:240] * 1e9)
    mean_feat_dict["q1_log_all"] = np.nanmean(q1_log_array, axis=0)
    mean_feat_dict["q2_log_all"] = np.nanmean(q2_log_array, axis=0)
    mean_feat_dict["q3_log_all"] = np.nanmean(q3_log_array, axis=0)
    std_feat_dict["q1_log_all"] = np.nanstd(q1_log_array, axis=0)
    std_feat_dict["q2_log_all"] = np.nanstd(q2_log_array, axis=0)
    std_feat_dict["q3_log_all"] = np.nanstd(q3_log_array, axis=0)
    mean_feat_dict["q1_log_all"] = np.nanmean(q1_log_array)
    mean_feat_dict["q2_log_all"] = np.nanmean(q2_log_array)
    mean_feat_dict["q3_log_all"] = np.nanmean(q3_log_array)
    std_feat_dict["q1_log_all"] = np.nanstd(q1_log_array)
    std_feat_dict["q2_log_all"] = np.nanstd(q2_log_array)
    std_feat_dict["q3_log_all"] = np.nanstd(q3_log_array)
    del q1_log_array, q2_log_array, q3_log_array
    gc.collect()

    # cloud_water_all, cloud_water_log_all
    cloud_water_array = x_array[:, 120:180] + x_array[:, 180:240]
    cloud_water_log_array = np.log1p(cloud_water_array * 1e9)
    mean_feat_dict["cloud_water"] = np.nanmean(cloud_water_array, axis=0)
    mean_feat_dict["cloud_water_log"] = np.nanmean(cloud_water_log_array, axis=0)
    std_feat_dict["cloud_water"] = np.nanstd(cloud_water_array, axis=0)
    std_feat_dict["cloud_water_log"] = np.nanstd(cloud_water_log_array, axis=0)
    mean_feat_dict["cloud_water_all"] = np.nanmean(cloud_water_array)
    mean_feat_dict["cloud_water_log_all"] = np.nanmean(cloud_water_log_array)
    std_feat_dict["cloud_water_all"] = np.nanstd(cloud_water_array)
    std_feat_dict["cloud_water_log_all"] = np.nanstd(cloud_water_log_array)

    del cloud_water_array, cloud_water_log_array
    gc.collect()

    # 特徴量の平均値
    q2q3_mean_array = (x_array[:, 120:180] + x_array[:, 180:240]) / 2
    uv_mean_array = (x_array[:, 240:300] + x_array[:, 300:360]) / 2
    pbuf_mean_array = (
        x_array[:, 376 : 376 + 60]
        + x_array[:, 376 + 60 : 376 + 120]
        + x_array[:, 376 + 120 : 376 + 180]
    ) / 3
    mean_feat_dict["q2q3_mean"] = np.nanmean(q2q3_mean_array, axis=0)
    mean_feat_dict["uv_mean"] = np.nanmean(uv_mean_array, axis=0)
    mean_feat_dict["pbuf_mean"] = np.nanmean(pbuf_mean_array, axis=0)
    std_feat_dict["q2q3_mean"] = np.nanstd(q2q3_mean_array, axis=0)
    std_feat_dict["uv_mean"] = np.nanstd(uv_mean_array, axis=0)
    std_feat_dict["pbuf_mean"] = np.nanstd(pbuf_mean_array, axis=0)

    # 下との差分
    t_diff_array = np.diff(
        x_array[:, 0:60], axis=1, append=0
    )  # 地上に近い方からの温度差を入れる
    q1_diff_array = np.diff(x_array[:, 60:120], axis=1, append=0)
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
    mean_feat_dict["t_diff"] = np.nanmean(t_diff_array, axis=0)
    mean_feat_dict["q1_diff"] = np.nanmean(q1_diff_array, axis=0)
    mean_feat_dict["q2_diff"] = np.nanmean(q2_diff_array, axis=0)
    mean_feat_dict["q3_diff"] = np.nanmean(q3_diff_array, axis=0)
    mean_feat_dict["u_diff"] = np.nanmean(u_diff_array, axis=0)
    mean_feat_dict["v_diff"] = np.nanmean(v_diff_array, axis=0)
    mean_feat_dict["ozone_diff"] = np.nanmean(ozone_diff_array, axis=0)
    mean_feat_dict["ch4_diff"] = np.nanmean(ch4_diff_array, axis=0)
    mean_feat_dict["n2o_diff"] = np.nanmean(n2o_diff_array, axis=0)
    mean_feat_dict["q2q3_mean_diff"] = np.nanmean(q2q3_mean_array_diff, axis=0)
    mean_feat_dict["uv_mean_diff"] = np.nanmean(uv_mean_array_diff, axis=0)
    mean_feat_dict["pbuf_mean_diff"] = np.nanmean(pbuf_mean_array_diff, axis=0)
    std_feat_dict["t_diff"] = np.nanstd(t_diff_array, axis=0)
    std_feat_dict["q1_diff"] = np.nanstd(q1_diff_array, axis=0)
    std_feat_dict["q2_diff"] = np.nanstd(q2_diff_array, axis=0)
    std_feat_dict["q3_diff"] = np.nanstd(q3_diff_array, axis=0)
    std_feat_dict["u_diff"] = np.nanstd(u_diff_array, axis=0)
    std_feat_dict["v_diff"] = np.nanstd(v_diff_array, axis=0)
    std_feat_dict["ozone_diff"] = np.nanstd(ozone_diff_array, axis=0)
    std_feat_dict["ch4_diff"] = np.nanstd(ch4_diff_array, axis=0)
    std_feat_dict["n2o_diff"] = np.nanstd(n2o_diff_array, axis=0)
    std_feat_dict["q2q3_mean_diff"] = np.nanstd(q2q3_mean_array_diff, axis=0)
    std_feat_dict["uv_mean_diff"] = np.nanstd(uv_mean_array_diff, axis=0)
    std_feat_dict["pbuf_mean_diff"] = np.nanstd(pbuf_mean_array_diff, axis=0)

    eps = 1e-60
    t_per_change_array = t_diff_array / (x_array[:, 0:60] + eps)
    q1_per_change_array = q1_diff_array / (x_array[:, 60:120] + eps)
    q2_per_change_array = q2_diff_array / (x_array[:, 120:180] + eps)
    q3_per_change_array = q3_diff_array / (x_array[:, 180:240] + eps)
    u_per_change_array = u_diff_array / (x_array[:, 240:300] + eps)
    v_per_change_array = v_diff_array / (x_array[:, 300:360] + eps)
    ozone_per_change_array = ozone_diff_array / (x_array[:, 376 : 376 + 60] + eps)
    ch4_per_change_array = ch4_diff_array / (x_array[:, 376 + 60 : 376 + 120] + eps)
    n2o_per_change_array = n2o_diff_array / (x_array[:, 376 + 120 : 376 + 180] + eps)
    q2q3_mean_array_per_change = q2q3_mean_array_diff / (q2q3_mean_array + eps)
    uv_mean_array_per_change = uv_mean_array_diff / (uv_mean_array + eps)
    pbuf_mean_array_per_change = pbuf_mean_array_diff / (pbuf_mean_array + eps)
    mean_feat_dict["t_per_change"] = np.nanmean(t_per_change_array, axis=0)
    mean_feat_dict["q1_per_change"] = np.nanmean(q1_per_change_array, axis=0)
    mean_feat_dict["q2_per_change"] = np.nanmean(q2_per_change_array, axis=0)
    mean_feat_dict["q3_per_change"] = np.nanmean(q3_per_change_array, axis=0)
    mean_feat_dict["u_per_change"] = np.nanmean(u_per_change_array, axis=0)
    mean_feat_dict["v_per_change"] = np.nanmean(v_per_change_array, axis=0)
    mean_feat_dict["ozone_per_change"] = np.nanmean(ozone_per_change_array, axis=0)
    mean_feat_dict["ch4_per_change"] = np.nanmean(ch4_per_change_array, axis=0)
    mean_feat_dict["n2o_per_change"] = np.nanmean(n2o_per_change_array, axis=0)
    mean_feat_dict["q2q3_mean_per_change"] = np.nanmean(
        q2q3_mean_array_per_change, axis=0
    )
    mean_feat_dict["uv_mean_per_change"] = np.nanmean(uv_mean_array_per_change, axis=0)
    mean_feat_dict["pbuf_mean_per_change"] = np.nanmean(
        pbuf_mean_array_per_change, axis=0
    )
    std_feat_dict["t_per_change"] = np.nanstd(t_per_change_array, axis=0)
    std_feat_dict["q1_per_change"] = np.nanstd(q1_per_change_array, axis=0)
    std_feat_dict["q2_per_change"] = np.nanstd(q2_per_change_array, axis=0)
    std_feat_dict["q3_per_change"] = np.nanstd(q3_per_change_array, axis=0)
    std_feat_dict["u_per_change"] = np.nanstd(u_per_change_array, axis=0)
    std_feat_dict["v_per_change"] = np.nanstd(v_per_change_array, axis=0)
    std_feat_dict["ozone_per_change"] = np.nanstd(ozone_per_change_array, axis=0)
    std_feat_dict["ch4_per_change"] = np.nanstd(ch4_per_change_array, axis=0)
    std_feat_dict["n2o_per_change"] = np.nanstd(n2o_per_change_array, axis=0)
    std_feat_dict["q2q3_mean_per_change"] = np.nanstd(
        q2q3_mean_array_per_change, axis=0
    )
    std_feat_dict["uv_mean_per_change"] = np.nanstd(uv_mean_array_per_change, axis=0)
    std_feat_dict["pbuf_mean_per_change"] = np.nanstd(
        pbuf_mean_array_per_change, axis=0
    )

    """
    # 差分のmean,stdを保存
    for window_size in [3, 5, 7]:
        print(f"window_size: {window_size}")
        (
            t_diff_array_mean_mean,
            t_diff_array_mean_std,
            t_diff_array_std_mean,
            t_diff_array_std_std,
        ) = rolling_mean_std(t_diff_array, window_size)
        (
            q1_diff_array_mean_mean,
            q1_diff_array_mean_std,
            q1_diff_array_std_mean,
            q1_diff_array_std_std,
        ) = rolling_mean_std(q1_diff_array, window_size)
        (
            q2_diff_array_mean_mean,
            q2_diff_array_mean_std,
            q2_diff_array_std_mean,
            q2_diff_array_std_std,
        ) = rolling_mean_std(q2_diff_array, window_size)
        (
            q3_diff_array_mean_mean,
            q3_diff_array_mean_std,
            q3_diff_array_std_mean,
            q3_diff_array_std_std,
        ) = rolling_mean_std(q3_diff_array, window_size)
        (
            u_diff_array_mean_mean,
            u_diff_array_mean_std,
            u_diff_array_std_mean,
            u_diff_array_std_std,
        ) = rolling_mean_std(u_diff_array, window_size)
        (
            v_diff_array_mean_mean,
            v_diff_array_mean_std,
            v_diff_array_std_mean,
            v_diff_array_std_std,
        ) = rolling_mean_std(v_diff_array, window_size)
        (
            ozone_diff_array_mean_mean,
            ozone_diff_array_mean_std,
            ozone_diff_array_std_mean,
            ozone_diff_array_std_std,
        ) = rolling_mean_std(ozone_diff_array, window_size)
        (
            ch4_diff_array_mean_mean,
            ch4_diff_array_mean_std,
            ch4_diff_array_std_mean,
            ch4_diff_array_std_std,
        ) = rolling_mean_std(ch4_diff_array, window_size)
        (
            n2o_diff_array_mean_mean,
            n2o_diff_array_mean_std,
            n2o_diff_array_std_mean,
            n2o_diff_array_std_std,
        ) = rolling_mean_std(n2o_diff_array, window_size)
        (
            q2q3_mean_array_diff_mean_mean,
            q2q3_mean_array_diff_mean_std,
            q2q3_mean_array_diff_std_mean,
            q2q3_mean_array_diff_std_std,
        ) = rolling_mean_std(q2q3_mean_array_diff, window_size)
        (
            uv_mean_array_diff_mean_mean,
            uv_mean_array_diff_mean_std,
            uv_mean_array_diff_std_mean,
            uv_mean_array_diff_std_std,
        ) = rolling_mean_std(uv_mean_array_diff, window_size)
        (
            pbuf_mean_array_diff_mean_mean,
            pbuf_mean_array_diff_mean_std,
            pbuf_mean_array_diff_std_mean,
            pbuf_mean_array_diff_std_std,
        ) = rolling_mean_std(pbuf_mean_array_diff, window_size)

        # mean mean
        mean_feat_dict[f"t_diff_mean_{window_size}"] = t_diff_array_mean_mean
        mean_feat_dict[f"q1_diff_mean_{window_size}"] = q1_diff_array_mean_mean
        mean_feat_dict[f"q2_diff_mean_{window_size}"] = q2_diff_array_mean_mean
        mean_feat_dict[f"q3_diff_mean_{window_size}"] = q3_diff_array_mean_mean
        mean_feat_dict[f"u_diff_mean_{window_size}"] = u_diff_array_mean_mean
        mean_feat_dict[f"v_diff_mean_{window_size}"] = v_diff_array_mean_mean
        mean_feat_dict[f"ozone_diff_mean_{window_size}"] = ozone_diff_array_mean_mean
        mean_feat_dict[f"ch4_diff_mean_{window_size}"] = ch4_diff_array_mean_mean
        mean_feat_dict[f"n2o_diff_mean_{window_size}"] = n2o_diff_array_mean_mean
        mean_feat_dict[f"q2q3_mean_diff_mean_{window_size}"] = (
            q2q3_mean_array_diff_mean_mean
        )
        mean_feat_dict[f"uv_mean_diff_mean_{window_size}"] = (
            uv_mean_array_diff_mean_mean
        )
        mean_feat_dict[f"pbuf_mean_diff_mean_{window_size}"] = (
            pbuf_mean_array_diff_mean_mean
        )

        # mean std
        std_feat_dict[f"t_diff_mean_{window_size}"] = t_diff_array_mean_std
        std_feat_dict[f"q1_diff_mean_{window_size}"] = q1_diff_array_mean_std
        std_feat_dict[f"q2_diff_mean_{window_size}"] = q2_diff_array_mean_std
        std_feat_dict[f"q3_diff_mean_{window_size}"] = q3_diff_array_mean_std
        std_feat_dict[f"u_diff_mean_{window_size}"] = u_diff_array_mean_std
        std_feat_dict[f"v_diff_mean_{window_size}"] = v_diff_array_mean_std
        std_feat_dict[f"ozone_diff_mean_{window_size}"] = ozone_diff_array_mean_std
        std_feat_dict[f"ch4_diff_mean_{window_size}"] = ch4_diff_array_mean_std
        std_feat_dict[f"n2o_diff_mean_{window_size}"] = n2o_diff_array_mean_std
        std_feat_dict[f"q2q3_mean_diff_mean_{window_size}"] = (
            q2q3_mean_array_diff_mean_std
        )
        std_feat_dict[f"uv_mean_diff_mean_{window_size}"] = uv_mean_array_diff_mean_std
        std_feat_dict[f"pbuf_mean_diff_mean_{window_size}"] = (
            pbuf_mean_array_diff_mean_std
        )

        # std mean
        mean_feat_dict[f"t_diff_std_{window_size}"] = t_diff_array_std_mean
        mean_feat_dict[f"q1_diff_std_{window_size}"] = q1_diff_array_std_mean
        mean_feat_dict[f"q2_diff_std_{window_size}"] = q2_diff_array_std_mean
        mean_feat_dict[f"q3_diff_std_{window_size}"] = q3_diff_array_std_mean
        mean_feat_dict[f"u_diff_std_{window_size}"] = u_diff_array_std_mean
        mean_feat_dict[f"v_diff_std_{window_size}"] = v_diff_array_std_mean
        mean_feat_dict[f"ozone_diff_std_{window_size}"] = ozone_diff_array_std_mean
        mean_feat_dict[f"ch4_diff_std_{window_size}"] = ch4_diff_array_std_mean
        mean_feat_dict[f"n2o_diff_std_{window_size}"] = n2o_diff_array_std_mean
        mean_feat_dict[f"q2q3_mean_diff_std_{window_size}"] = (
            q2q3_mean_array_diff_std_mean
        )
        mean_feat_dict[f"uv_mean_diff_std_{window_size}"] = uv_mean_array_diff_std_mean
        mean_feat_dict[f"pbuf_mean_diff_std_{window_size}"] = (
            pbuf_mean_array_diff_std_mean
        )

        # std std
        std_feat_dict[f"t_diff_std_{window_size}"] = t_diff_array_std_std
        std_feat_dict[f"q1_diff_std_{window_size}"] = q1_diff_array_std_std
        std_feat_dict[f"q2_diff_std_{window_size}"] = q2_diff_array_std_std
        std_feat_dict[f"q3_diff_std_{window_size}"] = q3_diff_array_std_std
        std_feat_dict[f"u_diff_std_{window_size}"] = u_diff_array_std_std
        std_feat_dict[f"v_diff_std_{window_size}"] = v_diff_array_std_std
        std_feat_dict[f"ozone_diff_std_{window_size}"] = ozone_diff_array_std_std
        std_feat_dict[f"ch4_diff_std_{window_size}"] = ch4_diff_array_std_std
        std_feat_dict[f"n2o_diff_std_{window_size}"] = n2o_diff_array_std_std
        std_feat_dict[f"q2q3_mean_diff_std_{window_size}"] = (
            q2q3_mean_array_diff_std_std
        )
        std_feat_dict[f"uv_mean_diff_std_{window_size}"] = uv_mean_array_diff_std_std
        std_feat_dict[f"pbuf_mean_diff_std_{window_size}"] = (
            pbuf_mean_array_diff_std_std
        )
    """
    del (
        t_diff_array,
        q1_diff_array,
        q2_diff_array,
        q3_diff_array,
        u_diff_array,
        v_diff_array,
    )
    del (
        ozone_diff_array,
        ch4_diff_array,
        n2o_diff_array,
        q2q3_mean_array_diff,
        uv_mean_array_diff,
        pbuf_mean_array_diff,
    )
    gc.collect()
    """
    # 上との差分
    t_diff_pre_array = np.diff(
        x_array[:, 0:60], axis=1, prepend=0
    )  # 上空に近い方からの温度差を入れる
    q1_diff_pre_array = np.diff(x_array[:, 60:120], axis=1, prepend=0)
    q2_diff_pre_array = np.diff(x_array[:, 120:180], axis=1, prepend=0)
    q3_diff_pre_array = np.diff(x_array[:, 180:240], axis=1, prepend=0)
    u_diff_pre_array = np.diff(x_array[:, 240:300], axis=1, prepend=0)
    v_diff_pre_array = np.diff(x_array[:, 300:360], axis=1, prepend=0)
    ozone_diff_pre_array = np.diff(x_array[:, 376 : 376 + 60], axis=1, prepend=0)
    ch4_diff_pre_array = np.diff(x_array[:, 376 + 60 : 376 + 120], axis=1, prepend=0)
    n2o_diff_pre_array = np.diff(x_array[:, 376 + 120 : 376 + 180], axis=1, prepend=0)
    q2q3_mean_array_diff_pre = np.diff(q2q3_mean_array, axis=1, prepend=0)
    uv_mean_array_diff_pre = np.diff(uv_mean_array, axis=1, prepend=0)
    pbuf_mean_array_diff_pre = np.diff(pbuf_mean_array, axis=1, prepend=0)
    mean_feat_dict["t_diff_pre"] = np.nanmean(t_diff_pre_array, axis=0)
    mean_feat_dict["q1_diff_pre"] = np.nanmean(q1_diff_pre_array, axis=0)
    mean_feat_dict["q2_diff_pre"] = np.nanmean(q2_diff_pre_array, axis=0)
    mean_feat_dict["q3_diff_pre"] = np.nanmean(q3_diff_pre_array, axis=0)
    mean_feat_dict["u_diff_pre"] = np.nanmean(u_diff_pre_array, axis=0)
    mean_feat_dict["v_diff_pre"] = np.nanmean(v_diff_pre_array, axis=0)
    mean_feat_dict["ozone_diff_pre"] = np.nanmean(ozone_diff_pre_array, axis=0)
    mean_feat_dict["ch4_diff_pre"] = np.nanmean(ch4_diff_pre_array, axis=0)
    mean_feat_dict["n2o_diff_pre"] = np.nanmean(n2o_diff_pre_array, axis=0)
    mean_feat_dict["q2q3_mean_diff_pre"] = np.nanmean(q2q3_mean_array_diff_pre, axis=0)
    mean_feat_dict["uv_mean_diff_pre"] = np.nanmean(uv_mean_array_diff_pre, axis=0)
    mean_feat_dict["pbuf_mean_diff_pre"] = np.nanmean(pbuf_mean_array_diff_pre, axis=0)
    std_feat_dict["t_diff_pre"] = np.nanstd(t_diff_pre_array, axis=0)
    std_feat_dict["q1_diff_pre"] = np.nanstd(q1_diff_pre_array, axis=0)
    std_feat_dict["q2_diff_pre"] = np.nanstd(q2_diff_pre_array, axis=0)
    std_feat_dict["q3_diff_pre"] = np.nanstd(q3_diff_pre_array, axis=0)
    std_feat_dict["u_diff_pre"] = np.nanstd(u_diff_pre_array, axis=0)
    std_feat_dict["v_diff_pre"] = np.nanstd(v_diff_pre_array, axis=0)
    std_feat_dict["ozone_diff_pre"] = np.nanstd(ozone_diff_pre_array, axis=0)
    std_feat_dict["ch4_diff_pre"] = np.nanstd(ch4_diff_pre_array, axis=0)
    std_feat_dict["n2o_diff_pre"] = np.nanstd(n2o_diff_pre_array, axis=0)
    std_feat_dict["q2q3_mean_diff_pre"] = np.nanstd(q2q3_mean_array_diff_pre, axis=0)
    std_feat_dict["uv_mean_diff_pre"] = np.nanstd(uv_mean_array_diff_pre, axis=0)
    std_feat_dict["pbuf_mean_diff_pre"] = np.nanstd(pbuf_mean_array_diff_pre, axis=0)
    del t_diff_pre_array, q1_diff_pre_array, q2_diff_pre_array, q3_diff_pre_array
    del u_diff_pre_array, v_diff_pre_array, ozone_diff_pre_array, ch4_diff_pre_array
    del n2o_diff_pre_array, q2q3_mean_array_diff_pre, uv_mean_array_diff_pre
    del pbuf_mean_array_diff_pre
    gc.collect()
    """

    mean_feat_dict["t_all"] = np.nanmean(x_array[:, 0:60])
    mean_feat_dict["q1_all"] = np.nanmean(x_array[:, 60:120])
    mean_feat_dict["q2_all"] = np.nanmean(x_array[:, 120:180])
    mean_feat_dict["q3_all"] = np.nanmean(x_array[:, 180:240])
    mean_feat_dict["u_all"] = np.nanmean(x_array[:, 240:300])
    mean_feat_dict["v_all"] = np.nanmean(x_array[:, 300:360])
    mean_feat_dict["ozone_all"] = np.nanmean(x_array[:, 376 : 376 + 60])
    mean_feat_dict["ch4_all"] = np.nanmean(x_array[:, 376 + 60 : 376 + 120])
    mean_feat_dict["n2o_all"] = np.nanmean(x_array[:, 376 + 120 : 376 + 180])

    std_feat_dict["t_all"] = np.nanstd(x_array[:, 0:60])
    std_feat_dict["q1_all"] = np.nanstd(x_array[:, 60:120])
    std_feat_dict["q2_all"] = np.nanstd(x_array[:, 120:180])
    std_feat_dict["q3_all"] = np.nanstd(x_array[:, 180:240])
    std_feat_dict["u_all"] = np.nanstd(x_array[:, 240:300])
    std_feat_dict["v_all"] = np.nanstd(x_array[:, 300:360])
    std_feat_dict["ozone_all"] = np.nanstd(x_array[:, 376 : 376 + 60])
    std_feat_dict["ch4_all"] = np.nanstd(x_array[:, 376 + 60 : 376 + 120])
    std_feat_dict["n2o_all"] = np.nanstd(x_array[:, 376 + 120 : 376 + 180])

    with open(output_path / "x_mean_feat_dict.pkl", "wb") as f:
        pickle.dump(mean_feat_dict, f)
    with open(output_path / "x_std_feat_dict.pkl", "wb") as f:
        pickle.dump(std_feat_dict, f)


def cal_stats_x_y(df, output_path):
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

    calc_feat(cfg, df[:, 1:557].to_numpy(), df[:, 557:].to_numpy(), output_path)

    cal_stats_x_y(
        df,
        output_path,
    )


if __name__ == "__main__":
    main()
