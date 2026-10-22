"""Core functions for the MMA3001 CO2 reconstruction project."""

import json
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def unpack_sensor_row(json_string):
    """
    Extract environmental measurements from one JSON sensor record.

    Parameters
    ----------
    json_string : str
        JSON string containing environmental sensor measurements.

    Returns
    -------
    dict
        Extracted sensor variables and the measurement timestamp.
        Missing measurements are returned as NaN.
    """
    data = json.loads(json_string)

    output = {}
    measurement_times = []

    for item in data:
        variable_name = item["variable"]["name"]
        values = item.get("values", [])

        if len(values) > 0:
            output[variable_name] = values[0].get("value", np.nan)

            if "time" in values[0]:
                measurement_times.append(values[0]["time"])

    if measurement_times:
        output["measurement_time"] = measurement_times[0]

    return output


def linear_reconstruct(timestamps, values, start, end):
    """
    Reconstruct a missing region using time-based linear interpolation.

    Parameters
    ----------
    timestamps : pandas.Series
        Datetime timestamps corresponding to the CO2 observations.
    values : array-like
        Original CO2 measurements.
    start : int
        Index position of the first missing observation.
    end : int
        Index position of the final missing observation.

    Returns
    -------
    numpy.ndarray
        Reconstructed values within the missing region.
    """
    series = pd.Series(
        np.asarray(values, dtype=float),
        index=pd.DatetimeIndex(timestamps)
    )

    masked = series.copy()
    masked.iloc[start:end + 1] = np.nan

    reconstructed = masked.interpolate(
        method="time",
        limit_area="inside"
    )

    return reconstructed.iloc[start:end + 1].to_numpy()


def pchip_reconstruct(timestamps, values, start, end):
    """
    Reconstruct a missing region using PCHIP interpolation.

    Parameters
    ----------
    timestamps : pandas.Series
        Datetime timestamps corresponding to the CO2 observations.
    values : array-like
        Original CO2 measurements.
    start : int
        Index position of the first missing observation.
    end : int
        Index position of the final missing observation.

    Returns
    -------
    numpy.ndarray
        PCHIP estimates within the missing region.
    """
    values = np.asarray(values, dtype=float)

    time_seconds = (
        pd.Series(timestamps).reset_index(drop=True)
        - pd.Series(timestamps).reset_index(drop=True).iloc[0]
    ).dt.total_seconds().to_numpy()

    known_mask = np.ones(len(values), dtype=bool)
    known_mask[start:end + 1] = False

    interpolator = PchipInterpolator(
        time_seconds[known_mask],
        values[known_mask],
        extrapolate=False
    )

    return interpolator(
        time_seconds[start:end + 1]
    )


def calculate_regression_metrics(true_values, predicted_values):
    """
    Calculate MAE, RMSE and R-squared.

    Parameters
    ----------
    true_values : array-like
        Ground-truth measurements.
    predicted_values : array-like
        Reconstructed or predicted measurements.

    Returns
    -------
    dict
        Dictionary containing MAE, RMSE and R2.
    """
    true_values = np.asarray(true_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)

    errors = true_values - predicted_values

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum(
        (true_values - np.mean(true_values)) ** 2
    )

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - ss_res / ss_tot

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
