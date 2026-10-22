
import sys
import numpy as np
import pandas as pd

sys.path.append("/content/src")

from reconstruction import (
    unpack_sensor_row,
    linear_reconstruct,
    pchip_reconstruct,
    calculate_regression_metrics
)


def test_unpack_sensor_row():
    """
    Check that valid JSON sensor data is unpacked correctly.
    """
    example_json = """
    [
        {
            "variable": {"name": "Carbon dioxide"},
            "values": [{"time": 1000, "value": 450}]
        },
        {
            "variable": {"name": "Temperature"},
            "values": [{"time": 1000, "value": 22.5}]
        }
    ]
    """

    result = unpack_sensor_row(example_json)

    assert result["Carbon dioxide"] == 450
    assert result["Temperature"] == 22.5
    assert result["measurement_time"] == 1000


def test_unpack_sensor_row_missing_value():
    """
    Check that a missing sensor value is stored as NaN.
    """
    example_json = """
    [
        {
            "variable": {"name": "Formaldehyde"},
            "values": [{"time": 1000}]
        }
    ]
    """

    result = unpack_sensor_row(example_json)

    assert np.isnan(result["Formaldehyde"])


def test_linear_reconstruction_known_case():
    """
    Verify linear interpolation against an analytically known result.
    """
    timestamps = pd.date_range(
        "2024-01-01",
        periods=5,
        freq="10min"
    )

    values = np.array([
        400,
        410,
        420,
        430,
        440
    ])

    reconstructed = linear_reconstruct(
        timestamps,
        values,
        start=2,
        end=2
    )

    assert np.isclose(
        reconstructed[0],
        420
    )


def test_pchip_reconstruction_linear_data():
    """
    PCHIP should reproduce a perfectly linear dataset exactly.
    """
    timestamps = pd.date_range(
        "2024-01-01",
        periods=5,
        freq="10min"
    )

    values = np.array([
        400,
        410,
        420,
        430,
        440
    ])

    reconstructed = pchip_reconstruct(
        timestamps,
        values,
        start=2,
        end=2
    )

    assert np.isclose(
        reconstructed[0],
        420
    )


def test_regression_metrics_perfect_prediction():
    """
    Perfect predictions should produce zero error and R2 = 1.
    """
    true_values = np.array([
        400,
        450,
        500
    ])

    predicted_values = true_values.copy()

    metrics = calculate_regression_metrics(
        true_values,
        predicted_values
    )

    assert np.isclose(
        metrics["MAE"],
        0
    )

    assert np.isclose(
        metrics["RMSE"],
        0
    )

    assert np.isclose(
        metrics["R2"],
        1
    )
