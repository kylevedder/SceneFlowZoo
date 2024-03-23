from bucketed_scene_flow_eval.utils.loaders import load_feather, load_json
from pathlib import Path
import pytest
import numpy as np


def test_validate_save_file():
    # Load the feather file
    file_folder = Path(
        "/tmp/argoverse2_tiny/val_constant_baseline_out/sequence_len_002/02678d04-cc9f-3148-9f95-1ba66347dff9/"
    )
    assert file_folder.exists(), f"Folder {file_folder} does not exist."
    feather_children = list(file_folder.glob("*.feather"))
    assert len(feather_children) == 1, f"Expected 1 feather file, got {len(feather_children)}"
    feather_file = feather_children[0]
    df = load_feather(feather_file)

    # Ensure that it has the following structure:
    #        is_valid  flow_tx_m  flow_ty_m  flow_tz_m
    # 0          True        1.0        2.0        3.0
    # 1          True        1.0        2.0        3.0
    # 2          True        1.0        2.0        3.0
    # 3          True        1.0        2.0        3.0
    # 4          True        1.0        2.0        3.0
    # ...         ...        ...        ...        ...
    # 90425      True        1.0        2.0        3.0
    # 90426      True        1.0        2.0        3.0
    # 90427      True        1.0        2.0        3.0
    # 90428      True        1.0        2.0        3.0
    # 90429      True        1.0        2.0        3.0

    assert len(df) == 90430, f"Expected 90430 rows, got {len(df)}"

    # Ensure df["is_valid"] is boolean
    assert (
        df["is_valid"].dtype == "bool"
    ), f"Expected is_valid to be boolean, got {df['is_valid'].dtype}"

    # Ensure that exactly 65225 of the is_valid are true
    assert df["is_valid"].sum() == 65225, f"Expected 65225 valid values, got {df['is_valid'].sum()}"

    # For valid rows, check if flow_tx_m, flow_ty_m, and flow_tz_m are 1.0, 2.0, and 3.0 respectively
    valid_flows = df[df["is_valid"]]
    assert all(
        valid_flows["flow_tx_m"] == 1.0
    ), f"Not all valid rows have flow_tx_m == 1.0; unique values: {valid_flows['flow_tx_m'].unique()}"
    assert all(
        valid_flows["flow_ty_m"] == 2.0
    ), f"Not all valid rows have flow_ty_m == 2.0, unique values: {valid_flows['flow_ty_m'].unique()}"
    assert all(
        valid_flows["flow_tz_m"] == 3.0
    ), f"Not all valid rows have flow_tz_m == 3.0, unique values: {valid_flows['flow_tz_m'].unique()}"

    # For invalid rows, check if flow_tx_m, flow_ty_m, and flow_tz_m are 0.0, 0.0, and 0.0 respectively
    invalid_flows = df[~df["is_valid"]]
    assert all(
        invalid_flows["flow_tx_m"] == 0.0
    ), f"Not all invalid rows have flow_tx_m == 0.0; unique values: {invalid_flows['flow_tx_m'].unique()}"
    assert all(
        invalid_flows["flow_ty_m"] == 0.0
    ), f"Not all invalid rows have flow_ty_m == 0.0; unique values: {invalid_flows['flow_ty_m'].unique()}"
    assert all(
        invalid_flows["flow_tz_m"] == 0.0
    ), f"Not all invalid rows have flow_tz_m == 0.0; unique values: {invalid_flows['flow_tz_m'].unique()}"


def _value_str_to_tuple(value: str) -> tuple[float, float]:
    # Convert nans and drop the parentheses
    return tuple(float(v) if v != "nan" else float("nan") for v in value.strip()[1:-1].split(", "))


def test_per_class_35():
    per_class_results = Path("/tmp/frame_results/bucketed_epe/per_class_results_35.json")
    assert per_class_results.exists(), f"File {per_class_results} does not exist."

    # Load the json file
    results = load_json(per_class_results)

    EXPECTED_RESULTS = {
        "BACKGROUND": "(3.741657, nan)",
        "CAR": "(3.740714, 10.937946)",
        "OTHER_VEHICLES": "(nan, nan)",
        "PEDESTRIAN": "(nan, 27.405553)",
        "WHEELED_VRU": "(nan, 11.003660)",
    }

    # Check matching keys
    assert set(results.keys()) == set(
        EXPECTED_RESULTS.keys()
    ), f"Expected keys {set(EXPECTED_RESULTS.keys())}, got {set(results.keys())}"

    # Ensure that the results are as expected
    for class_name, expected_result in EXPECTED_RESULTS.items():
        exp_static, exp_dyn = _value_str_to_tuple(expected_result)
        extr_static, extr_dyn = _value_str_to_tuple(results[class_name])
        assert np.isnan(exp_static) == np.isnan(
            extr_static
        ), f"Expected {exp_static}, got {extr_static}"
        assert np.isnan(exp_dyn) == np.isnan(extr_dyn), f"Expected {exp_dyn}, got {extr_dyn}"

        if not np.isnan(exp_static):
            assert (
                pytest.approx(extr_static) == exp_static
            ), f"Expected {exp_static}, got {extr_static}"

        if not np.isnan(exp_dyn):
            assert pytest.approx(extr_dyn) == exp_dyn, f"Expected {exp_dyn}, got {extr_dyn}"


def test_per_class_inf():
    per_class_results = Path("/tmp/frame_results/bucketed_epe/per_class_results_inf.json")
    assert per_class_results.exists(), f"File {per_class_results} does not exist."

    # Load the json file
    results = load_json(per_class_results)

    EXPECTED_RESULTS = {
        "BACKGROUND": "(3.741657, nan)",
        "CAR": "(3.740686, 25.886737)",
        "OTHER_VEHICLES": "(nan, nan)",
        "PEDESTRIAN": "(3.739319, 26.824205)",
        "WHEELED_VRU": "(nan, 11.003660)",
    }

    # Check matching keys
    assert set(results.keys()) == set(
        EXPECTED_RESULTS.keys()
    ), f"Expected keys {set(EXPECTED_RESULTS.keys())}, got {set(results.keys())}"

    # Ensure that the results are as expected
    for class_name, expected_result in EXPECTED_RESULTS.items():
        exp_static, exp_dyn = _value_str_to_tuple(expected_result)
        extr_static, extr_dyn = _value_str_to_tuple(results[class_name])
        assert np.isnan(exp_static) == np.isnan(
            extr_static
        ), f"Expected {exp_static}, got {extr_static}"
        assert np.isnan(exp_dyn) == np.isnan(extr_dyn), f"Expected {exp_dyn}, got {extr_dyn}"

        if not np.isnan(exp_static):
            assert (
                pytest.approx(extr_static) == exp_static
            ), f"Expected {exp_static}, got {extr_static}"

        if not np.isnan(exp_dyn):
            assert (
                pytest.approx(extr_dyn, abs=4e-5) == exp_dyn
            ), f"Expected {exp_dyn}, got {extr_dyn}"


def test_mean_average_35():
    # Test /tmp/frame_results/bucketed_epe/mean_average_results_35.json
    mean_average_results = Path("/tmp/frame_results/bucketed_epe/mean_average_results_35.json")
    assert mean_average_results.exists(), f"File {mean_average_results} does not exist."

    # Load the json file
    results = load_json(mean_average_results)

    EXPECTED_RESULTS = [3.7411858173458397, 16.449052669388]

    assert pytest.approx(results) == EXPECTED_RESULTS, f"Expected {EXPECTED_RESULTS}, got {results}"


def test_mean_average_inf():
    # Test /tmp/frame_results/bucketed_epe/mean_average_results_inf.json
    mean_average_results = Path("/tmp/frame_results/bucketed_epe/mean_average_results_inf.json")
    assert mean_average_results.exists(), f"File {mean_average_results} does not exist."

    # Load the json file
    results = load_json(mean_average_results)

    EXPECTED_RESULTS = [3.74055435191852, 21.23820050931167]

    # assert results == EXPECTED_RESULTS, f"Expected {EXPECTED_RESULTS}, got {results}"
    # Use pytest to compare floating point numbers
    assert pytest.approx(results) == EXPECTED_RESULTS, f"Expected {EXPECTED_RESULTS}, got {results}"
