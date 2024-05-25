import numpy as np


def eliq(T, method="paper"):
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
    elif method == "tenten":
        # https://metview.readthedocs.io/en/latest/api/functions/saturation_vapour_pressure.html
        a1 = 611.21
        a3 = 17.502
        a4 = 32.19
        return a1 * np.exp(a3 * (T - 273.16) / (T - a4))


def eice(T, method="paper"):
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
            (T > c_ice[0]) * eliq(T)
            + (T <= c_ice[0]) * (T > c_ice[1]) * 100 * np.polyval(a_ice, T - T0)
            + (T <= c_ice[1])
            * 100
            * (
                c_ice[3]
                + np.maximum(c_ice[2], T - T0)
                * (c_ice[4] + np.maximum(c_ice[2], T - T0) * c_ice[5])
            )
        )
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
    method="paper",
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
