import numpy as np
import scipy.integrate as sin

L_V = 2.501e6  # Latent heat of vaporization
L_I = 3.337e5  # Latent heat of freezing
L_F = L_I
L_S = L_V + L_I  # Sublimation
C_P = 1.00464e3  # Specific heat capacity of air at constant pressure
G = 9.80616
RHO_L = 1e3


class ThermLibNumpy:
    @staticmethod
    def eliqNumpy(T):
        a_liq = np.float32(
            np.array(
                [
                    -0.976195544e-15,
                    -0.952447341e-13,
                    0.640689451e-10,
                    0.206739458e-7,
                    0.302950461e-5,
                    0.264847430e-3,
                    0.142986287e-1,
                    0.443987641,
                    6.11239921,
                ]
            )
        )
        c_liq = np.float32(-80.0)
        T0 = np.float32(273.16)
        return np.float32(100.0) * np.polyval(a_liq, np.maximum(c_liq, T - T0))

    @staticmethod
    def eiceNumpy(T):
        a_ice = np.float32(
            np.array(
                [
                    0.252751365e-14,
                    0.146898966e-11,
                    0.385852041e-9,
                    0.602588177e-7,
                    0.615021634e-5,
                    0.420895665e-3,
                    0.188439774e-1,
                    0.503160820,
                    6.11147274,
                ]
            )
        )
        c_ice = np.float32(
            np.array([273.15, 185, -100, 0.00763685, 0.000151069, 7.48215e-07])
        )
        T0 = np.float32(273.16)
        return np.where(
            T > c_ice[0],
            ThermLibNumpy.eliqNumpy(T),
            np.where(
                T <= c_ice[1],
                np.float32(100.0)
                * (
                    c_ice[3]
                    + np.maximum(c_ice[2], T - T0)
                    * (c_ice[4] + np.maximum(c_ice[2], T - T0) * c_ice[5])
                ),
                np.float32(100.0) * np.polyval(a_ice, T - T0),
            ),
        )

    @staticmethod
    def esatNumpy(T):
        T0 = np.float32(273.16)
        T00 = np.float32(253.16)
        omtmp = (T - T00) / (T0 - T00)
        omega = np.maximum(np.float32(0.0), np.minimum(np.float32(1.0), omtmp))

        return np.where(
            T > T0,
            ThermLibNumpy.eliqNumpy(T),
            np.where(
                T < T00,
                ThermLibNumpy.eiceNumpy(T),
                (
                    omega * ThermLibNumpy.eliqNumpy(T)
                    + (1 - omega) * ThermLibNumpy.eiceNumpy(T)
                ),
            ),
        )

    @staticmethod
    def qvNumpy(T, RH, P0, PS, hyam, hybm):
        R = np.float32(287.0)
        Rv = np.float32(461.0)
        p = P0 * hyam + PS[:, None] * hybm  # Total pressure (Pa)

        T = T.astype(np.float32)
        if type(RH) == int:
            RH = T**0
        RH = RH.astype(np.float32)
        p = p.astype(np.float32)

        return R * ThermLibNumpy.esatNumpy(T) * RH / (Rv * p)

    @staticmethod
    def RHNumpy(T, qv, P0, PS, hyam, hybm):
        R = np.float32(287.0)
        Rv = np.float32(461.0)
        p = P0 * hyam + PS[:, None] * hybm  # Total pressure (Pa)

        T = T.astype(np.float32)
        qv = qv.astype(np.float32)
        p = p.astype(np.float32)

        return Rv * p * qv / (R * ThermLibNumpy.esatNumpy(T))

    @staticmethod
    def qsatNumpy(T, P0, PS, hyam, hybm):
        return ThermLibNumpy.qvNumpy(T, 1, P0, PS, hyam, hybm)

    @staticmethod
    def qsatsurfNumpy(TS, P0, PS):
        R = 287
        Rv = 461
        return R * ThermLibNumpy.esatNumpy(TS) / (Rv * PS)

    @staticmethod
    def theta_e_calc(T, qv, P0, PS, hyam, hybm):
        S = PS.shape
        p = P0 * np.tile(hyam, (S[0], 1)) + np.transpose(
            np.tile(PS, (60, 1))
        ) * np.tile(hybm, (S[0], 1))
        RV = 461.5
        RD = 287.04
        EPS = RD / RV
        r = qv / (1.0 - qv)
        # get ev in hPa
        ev_hPa = 100 * p * r / (EPS + r)
        # get TL
        TL = (2840.0 / ((3.5 * np.log(T)) - (np.log(ev_hPa)) - 4.805)) + 55.0
        # calc chi_e:
        chi_e = 0.2854 * (1.0 - (0.28 * r))
        P0_norm = P0 / (
            P0 * np.tile(hyam, (S[0], 1))
            + np.transpose(np.tile(PS, (60, 1))) * np.tile(hybm, (S[0], 1))
        )
        theta_e = (
            T
            * P0_norm**chi_e
            * np.exp(((3.376 / TL) - 0.00254) * r * 1000.0 * (1.0 + (0.81 * r)))
        )
        return theta_e

    @staticmethod
    def theta_e_sat_calc(T, P0, PS, hyam, hybm):
        return ThermLibNumpy.theta_e_calc(
            T, ThermLibNumpy.qsatNumpy(T, P0, PS, hyam, hybm), P0, PS, hyam, hybm
        )

    @staticmethod
    def bmse_calc(T, qv, P0, PS, hyam, hybm):
        eps = 0.622  # Ratio of molecular weight(H2O)/molecular weight(dry air)
        R_D = 287  # Specific gas constant of dry air in J/K/kg
        Rv = 461
        # Calculate kappa
        QSAT0 = ThermLibNumpy.qsatNumpy(T, P0, PS, hyam, hybm)
        kappa = 1 + (L_V**2) * QSAT0 / (Rv * C_P * (T**2))
        # Calculate geopotential
        r = qv / (qv**0 - qv)
        Tv = T * (r**0 + r / eps) / (r**0 + r)
        p = P0 * hyam + PS[:, None] * hybm
        p = p.astype(np.float32)
        RHO = p / (R_D * Tv)
        Z = -sin.cumtrapz(x=p, y=1 / (G * RHO), axis=1)
        Z = np.concatenate((0 * Z[:, 0:1] ** 0, Z), axis=1)
        # Assuming near-surface is at 2 meters
        Z = (Z - Z[:, [29]]) + 2
        # Calculate MSEs of plume and environment
        Tile_dim = [1, 60]
        h_plume = np.tile(
            np.expand_dims(C_P * T[:, -1] + L_V * qv[:, -1], axis=1), Tile_dim
        )
        h_satenv = C_P * T + L_V * qv + G * Z
        return (G / kappa) * (h_plume - h_satenv) / (C_P * T)


def eliq(T, method="Bolton"):
    """
    Function taking temperature (in K) and outputting liquid saturation
    pressure (in hPa) using a polynomial fit
    """
    if method == "paper":
        a_liq = np.array(
            [
                -0.976195544e-15,
                -0.952447341e-13,
                0.640689451e-10,
                0.206739458e-7,
                0.302950461e-5,
                0.264847430e-3,
                0.142986287e-1,
                0.443987641,
                6.11239921,
            ]
        )
        c_liq = -80
        T0 = 273.16
        return 100 * np.polyval(a_liq, np.maximum(c_liq, T - T0))

    elif method == "MurphyKoop":
        es = np.exp(
            54.842763
            - (6763.22 / T)
            - (4.210 * np.log(T))
            + (0.000367 * T)
            + (
                np.tanh(0.0415 * (T - 218.8))
                * (53.878 - (1331.22 / T) - (9.44523 * np.log(T)) + 0.014025 * T)
            )
        )
        return es
    elif method == "GoffGratch":
        tboil = 373.15  # Boiling point of water in Kelvin
        es = (
            10.0
            ** (
                -7.90298 * (tboil / T - 1.0)
                + 5.02808 * np.log10(tboil / T)
                - 1.3816e-7 * (10.0 ** (11.344 * (1.0 - T / tboil)) - 1.0)
                + 8.1328e-3 * (10.0 ** (-3.49149 * (tboil / T - 1.0)) - 1.0)
                + np.log10(1013.246)
            )
            * 100.0
        )
        return es
    elif method == "OldGoffGratch":
        tboil = 373.15  # Boiling point of water in Kelvin
        ps = 1013.246
        e1 = 11.344 * (1.0 - T / tboil)
        e2 = -3.49149 * (tboil / T - 1.0)
        f1 = -7.90298 * (tboil / T - 1.0)
        f2 = 5.02808 * np.log10(tboil / T)
        f3 = -1.3816 * (10.0**e1 - 1.0) / 10000000.0
        f4 = 8.1328 * (10.0**e2 - 1.0) / 1000.0
        f5 = np.log10(ps)
        f = f1 + f2 + f3 + f4 + f5
        es = (10.0**f) * 100.0
        return es
    elif method == "e3sm":
        a0 = 6.105851
        a1 = 0.4440316
        a2 = 0.1430341e-1
        a3 = 0.2641412e-3
        a4 = 0.2995057e-5
        a5 = 0.2031998e-7
        a6 = 0.6936113e-10
        a7 = 0.2564861e-13
        a8 = -0.3704404e-15

        dtt = T - 273.16
        esatw = a0 + dtt * (
            a1
            + dtt
            * (
                a2
                + dtt
                * (a3 + dtt * (a4 + dtt * (a5 + dtt * (a6 + dtt * (a7 + a8 * dtt)))))
            )
        )

        index = dtt <= -80.0
        esatw[index] = (
            2.0
            * 0.01
            * np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T)
        )[index]
        return esatw * (10**2)
    elif method == "Bolton":
        c1 = 611.2
        c2 = 17.67
        c3 = 243.5
        tmelt = 273.15  # Melting point of water in Kelvin
        es = c1 * np.exp((c2 * (T - tmelt)) / ((T - tmelt) + c3))
        return es
    elif method == "tenten":
        # https://metview.readthedocs.io/en/latest/api/functions/saturation_vapour_pressure.html
        a1 = 611.21
        a3 = 17.502
        a4 = 32.19
        return a1 * np.exp(a3 * (T - 273.16) / (T - a4))


def eice(T, method="Bolton"):
    """
    Function taking temperature (in K) and outputting ice saturation
    pressure (in hPa) using a polynomial fit
    """
    if method == "paper":
        a_ice = np.array(
            [
                0.252751365e-14,
                0.146898966e-11,
                0.385852041e-9,
                0.602588177e-7,
                0.615021634e-5,
                0.420895665e-3,
                0.188439774e-1,
                0.503160820,
                6.11147274,
            ]
        )
        c_ice = np.array([273.15, 185, -100, 0.00763685, 0.000151069, 7.48215e-07])
        T0 = 273.16
        return (
            (T > c_ice[0]) * eliq(T, method)
            + (T <= c_ice[0]) * (T > c_ice[1]) * 100 * np.polyval(a_ice, T - T0)
            + (T <= c_ice[1])
            * 100
            * (
                c_ice[3]
                + np.maximum(c_ice[2], T - T0)
                * (c_ice[4] + np.maximum(c_ice[2], T - T0) * c_ice[5])
            )
        )
    elif method == "GoffGratch":
        h2otrip = 273.16  # H2O triple point temperature in Kelvin
        es = (
            10.0
            ** (
                -9.09718 * (h2otrip / T - 1.0)
                - 3.56654 * np.log10(h2otrip / T)
                + 0.876793 * (1.0 - T / h2otrip)
                + np.log10(6.1071)
            )
            * 100.0
        )
        return es
    elif method == "OldGoffGratch":
        tmelt = 273.15  # Melting point of ice in Kelvin
        term1 = 2.01889049 / (tmelt / T)
        term2 = 3.56654 * np.log(tmelt / T)
        term3 = 20.947031 * (tmelt / T)
        es = 575.185606e10 * np.exp(-(term1 + term2 + term3))
        return es
    elif method == "MurphyKoop":
        es = np.exp(
            9.550426 - (5723.265 / T) + (3.53068 * np.log(T)) - (0.00728332 * T)
        )
        return es
    elif method == "e3sm":
        a0 = 6.11147274
        a1 = 0.503160820
        a2 = 0.188439774e-1
        a3 = 0.420895665e-3
        a4 = 0.615021634e-5
        a5 = 0.602588177e-7
        a6 = 0.385852041e-9
        a7 = 0.146898966e-11
        a8 = 0.252751365e-14

        dtt = T - 273.16
        esati = a0 + dtt * (
            a1
            + dtt
            * (
                a2
                + dtt
                * (a3 + dtt * (a4 + dtt * (a5 + dtt * (a6 + dtt * (a7 + a8 * dtt)))))
            )
        )

        index = dtt <= -80.0
        esati[index] = (
            0.01
            * np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T)
        )[index]
        return esati * (10**2)
    elif method == "Bolton":
        c1 = 611.2
        c2 = 17.67
        c3 = 243.5
        tmelt = 273.15  # Melting point of water in Kelvin
        es = c1 * np.exp((c2 * (T - tmelt)) / ((T - tmelt) + c3))
        return es
    elif method == "tenten":
        # https://metview.readthedocs.io/en/latest/api/functions/saturation_vapour_pressure.html
        a1 = 611.21
        a3 = 22.587
        a4 = -0.7
        return a1 * np.exp(a3 * (T - 273.16) / (T - a4))


def cal_specific2relative_coef(
    temperature_array,
    near_surface_air_pressure,
    hyam,
    hybm,
    method="Bolton",
):
    """
    specific humidity を relative humidity に変換するための係数を算出する（逆数を取れば逆変換にも使える）
    """
    P0 = 1e5  # Mean surface air pressure (Pa)
    # Formula to calculate air pressure (in Pa) using the hybrid vertical grid
    # coefficients at the middle of each vertical level: hyam and hybm
    air_pressure_Pa = hyam * P0 + hybm[None, :] * near_surface_air_pressure[:, None]

    # 1) Calculating saturation water vapor pressure
    T0 = 273.16  # Freezing temperature in standard conditions
    T00 = 253.16  # Temperature below which we use e_ice
    omega = (temperature_array - T00) / (T0 - T00)
    omega = np.maximum(0, np.minimum(1, omega))

    esat = omega * eliq(temperature_array, method) + (1 - omega) * eice(
        temperature_array, method
    )
    # 2) Calculating relative humidity
    Rd = 287  # Specific gas constant for dry air
    Rv = 461  # Specific gas constant for water vapor

    # We use the `values` method to convert Xarray DataArray into Numpy ND-Arrays
    return Rv / Rd * air_pressure_Pa / esat


def cal_normalized_lfh(qvprior, Tprior, PSprior, LHFprior, hyam, hybm):
    P0 = 1e5
    epsilon = 1e-3
    Qdenprior = (ThermLibNumpy.qsatNumpy(Tprior, P0, PSprior, hyam, hybm))[
        :, -1
    ] - qvprior[:, -1]
    Qdenprior = np.maximum(epsilon, Qdenprior)

    Tile_dim = [1, 1]
    # LHFtile = np.tile(np.expand_dims(LHFprior,axis=1),Tile_dim)
    LHFscaled = LHFprior / (L_V * Qdenprior)
    LHFtile = np.tile(np.expand_dims(LHFscaled.astype(np.float32), axis=1), Tile_dim)

    return LHFtile
