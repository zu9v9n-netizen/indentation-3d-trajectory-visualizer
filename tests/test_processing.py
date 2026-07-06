from __future__ import annotations

import numpy as np
import pandas as pd

from indentation_3d_trajectory_visualizer import VisualizationConfig, process_dataframe


def make_sample_dataframe() -> pd.DataFrame:
    time = np.arange(0, 4.0, 0.01)
    load = np.zeros_like(time)
    displacement = np.zeros_like(time)

    active = (time >= 1.0) & (time <= 2.2)
    load[active] = np.sin((time[active] - 1.0) / 1.2 * np.pi) * 0.25
    displacement[active] = (time[active] - 1.0) * 0.2
    after = time > 2.2
    displacement[after] = 0.24

    return pd.DataFrame({"time_s": time, "load_N": load, "displacement_mm": displacement})


def test_process_dataframe_creates_contact_corrected_trial() -> None:
    config = VisualizationConfig(
        sample_interval_sec=0.01,
        smoothing_window_samples=5,
        pre_margin_sec=0.1,
        post_margin_sec=0.1,
        min_active_duration_sec=0.05,
    )
    corrected, contacts, _ = process_dataframe(
        make_sample_dataframe(),
        config,
        subject="sample",
        condition="room",
        source_csv="sample.csv",
    )

    assert len(contacts) == 1
    assert not corrected.empty
    assert corrected["trial_id"].nunique() == 1
    assert contacts.loc[0, "event_quality_label"] == "normal"

    contact_rows = corrected[corrected["global_sample"] == contacts.loc[0, "contact_start_index"]]
    assert abs(float(contact_rows.iloc[0]["indentation_mm"])) < 1e-9
    assert abs(float(contact_rows.iloc[0]["relative_time_s"])) < 1e-9

