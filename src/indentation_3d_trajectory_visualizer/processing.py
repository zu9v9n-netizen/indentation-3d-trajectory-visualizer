from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import VisualizationConfig

ENCODINGS_TO_TRY = ("utf-8-sig", "cp932", "shift_jis", "utf-8")


@dataclass
class CandidateRegion:
    region_id: int
    active_start_index: int
    active_end_index: int
    threshold_start_index: int
    threshold_end_index: int
    baseline_start_index: int
    baseline_end_index: int
    baseline_displacement_mm: float
    contact_start_index: int
    contact_load_offset_N: float
    contact_displacement_mm: float
    displacement_excursion_mm: float
    max_load_N: float
    small_displacement_load_event_flag: bool
    valid_indentation_event_flag: bool
    excluded_reason: str


@dataclass
class TrialRegion:
    trial_id: int
    region_id: int
    active_start_index: int
    active_end_index: int
    threshold_start_index: int
    threshold_end_index: int
    cut_start_index: int
    cut_end_index: int
    contact_start_index: int
    contact_load_offset_N: float
    contact_displacement_mm: float
    displacement_excursion_mm: float
    min_valid_displacement_excursion_mm: float
    small_displacement_load_event_flag: bool
    valid_indentation_event_flag: bool
    excluded_reason: str


def read_input_csv(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {path}")

    last_error: Exception | None = None
    for encoding in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
            if _headers_look_like_data(df.columns):
                df = pd.read_csv(path, sep=None, engine="python", encoding=encoding, header=None)
            if df.shape[1] <= 1:
                df = pd.read_csv(path, encoding=encoding)
                if _headers_look_like_data(df.columns):
                    df = pd.read_csv(path, encoding=encoding, header=None)
            return df, encoding
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read CSV with supported encodings: {last_error}")


def _headers_look_like_data(columns: pd.Index) -> bool:
    numeric_like = 0
    for col in columns:
        try:
            float(str(col).strip())
            numeric_like += 1
        except ValueError:
            pass
    return len(columns) > 0 and numeric_like / len(columns) >= 0.8


def _find_column_by_candidates(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | int | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def parse_column(value: str | None) -> str | int | None:
    if value is None or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def get_column_data(df: pd.DataFrame, column: str | int | None, role: str) -> pd.Series | None:
    candidates = {
        "time": ("time", "time_s", "timestamp", "sec", "seconds", "時刻", "時間"),
        "load": ("load", "load_n", "force", "force_n", "荷重", "力"),
        "displacement": (
            "displacement",
            "displacement_mm",
            "disp",
            "position",
            "変位",
            "押し込み",
        ),
    }
    selected: str | int | None = column
    if selected is None:
        selected = _find_column_by_candidates(df, candidates[role])

    if selected is None:
        if role == "load" and df.shape[1] >= 1:
            return df.iloc[:, 0]
        if role == "displacement" and df.shape[1] >= 2:
            return df.iloc[:, 1]
        return None

    if isinstance(selected, int):
        if selected < 0 or selected >= df.shape[1]:
            raise ValueError(f"{role} column index is out of range: {selected}")
        return df.iloc[:, selected]
    if selected not in df.columns:
        raise ValueError(f"{role} column was not found: {selected}")
    return df[selected]


def generate_time_column(length: int, sample_interval_sec: float) -> pd.Series:
    return pd.Series(np.arange(length, dtype=float) * sample_interval_sec)


def smooth_load(load: pd.Series, window_samples: int) -> pd.Series:
    window = max(1, int(window_samples))
    if window % 2 == 0:
        window += 1
    return load.rolling(window=window, center=True, min_periods=1).mean()


def estimate_detection_load_baseline(load_smooth: pd.Series) -> float:
    values = pd.to_numeric(load_smooth, errors="coerce").dropna()
    if values.empty:
        return 0.0
    low_limit = values.quantile(0.4)
    rest_like = values[values <= low_limit]
    baseline_source = rest_like if not rest_like.empty else values
    return float(baseline_source.median())


def preprocess_data(raw_df: pd.DataFrame, config: VisualizationConfig) -> pd.DataFrame:
    load = get_column_data(raw_df, config.load_column, "load")
    displacement = get_column_data(raw_df, config.displacement_column, "displacement")
    time = get_column_data(raw_df, config.time_column, "time")

    if load is None:
        raise ValueError("load column could not be resolved")
    if displacement is None:
        raise ValueError("displacement column could not be resolved")
    if time is None:
        time = generate_time_column(len(raw_df), config.sample_interval_sec)

    df = pd.DataFrame(
        {
            "raw_time_s": pd.to_numeric(time, errors="coerce"),
            "raw_load_N": pd.to_numeric(load, errors="coerce"),
            "raw_displacement_mm": pd.to_numeric(displacement, errors="coerce"),
        }
    ).dropna()
    df = df.reset_index(drop=True)
    df["global_sample"] = np.arange(len(df), dtype=int)
    df["time_s"] = df["raw_time_s"].astype(float)
    df["global_time_s"] = df["raw_time_s"].astype(float)
    df["load_N"] = df["raw_load_N"].astype(float)
    df["displacement_mm"] = df["raw_displacement_mm"].astype(float)
    df["load_smooth_N"] = smooth_load(df["load_N"], config.smoothing_window_samples)

    baseline = estimate_detection_load_baseline(df["load_smooth_N"])
    df["detection_load_offset_N"] = baseline
    df["load_detection_N"] = df["load_N"] - baseline
    df["load_smooth_detection_N"] = df["load_smooth_N"] - baseline
    return df


def detect_load_candidate_regions(df: pd.DataFrame, config: VisualizationConfig) -> list[tuple[int, int]]:
    active = df["load_smooth_detection_N"].to_numpy() > config.load_threshold_n
    if not active.any():
        return []

    gap_samples = max(0, int(round(config.max_gap_within_trial_sec / config.sample_interval_sec)))
    min_samples = max(1, int(round(config.min_active_duration_sec / config.sample_interval_sec)))
    pre_samples = max(0, int(round(config.pre_margin_sec / config.sample_interval_sec)))
    post_samples = max(0, int(round(config.post_margin_sec / config.sample_interval_sec)))

    ranges = _true_ranges(active)
    merged = _merge_short_gaps(ranges, gap_samples)

    return [(start, end) for start, end in merged if (end - start + 1) >= min_samples]


def _true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            ranges.append((start, idx - 1))
            start = None
    if start is not None:
        ranges.append((start, len(mask) - 1))
    return ranges


def _merge_short_gaps(ranges: list[tuple[int, int]], gap_samples: int) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= gap_samples:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def find_contact_start_index(df: pd.DataFrame, threshold_start_index: int, config: VisualizationConfig) -> int:
    contact_threshold = max(0.001, config.load_threshold_n * 0.02)
    load = df["load_detection_N"].to_numpy()
    pre_samples = max(0, int(round(config.pre_margin_sec / config.sample_interval_sec)))
    search_start = max(0, threshold_start_index - pre_samples)
    for idx in range(threshold_start_index - 1, search_start - 1, -1):
        value = load[idx]
        if np.isfinite(value) and value <= contact_threshold:
            return int(idx)
    return int(threshold_start_index)


def characterize_candidate_regions(
    df: pd.DataFrame,
    regions: list[tuple[int, int]],
    config: VisualizationConfig,
) -> list[CandidateRegion]:
    baseline_samples = int(round(config.pre_margin_sec / config.sample_interval_sec))
    candidates: list[CandidateRegion] = []
    for region_id, (start, end) in enumerate(regions, start=1):
        contact_start = find_contact_start_index(df, start, config)
        baseline_start = max(0, contact_start - baseline_samples)
        baseline_end = max(0, contact_start - 1)
        baseline = df.iloc[baseline_start : baseline_end + 1]
        if baseline.empty:
            baseline = df.iloc[[contact_start]]
            baseline_start = contact_start
            baseline_end = contact_start
        load_offset = float(baseline["raw_load_N"].median()) if config.zero_correction else 0.0
        contact_displacement = float(df.iloc[contact_start]["displacement_mm"]) if config.zero_correction else 0.0
        candidate = df.iloc[contact_start : end + 1]
        displacement_excursion = float((candidate["displacement_mm"] - contact_displacement).abs().max())
        max_load = float(candidate["load_smooth_detection_N"].max())
        small_event = displacement_excursion < config.min_valid_displacement_excursion_mm
        excluded_reason = "small_displacement_load_event" if small_event else ""
        candidates.append(
            CandidateRegion(
                region_id=region_id,
                active_start_index=contact_start,
                active_end_index=end,
                threshold_start_index=start,
                threshold_end_index=end,
                baseline_start_index=baseline_start,
                baseline_end_index=baseline_end,
                baseline_displacement_mm=contact_displacement,
                contact_start_index=contact_start,
                contact_load_offset_N=load_offset,
                contact_displacement_mm=contact_displacement,
                displacement_excursion_mm=displacement_excursion,
                max_load_N=max_load,
                small_displacement_load_event_flag=small_event,
                valid_indentation_event_flag=not small_event,
                excluded_reason=excluded_reason,
            )
        )
    return candidates


def create_trial_segments(
    regions: list[CandidateRegion], n_rows: int, config: VisualizationConfig
) -> list[TrialRegion]:
    pre = int(round(config.pre_margin_sec / config.sample_interval_sec))
    post = int(round(config.post_margin_sec / config.sample_interval_sec))
    return [
        TrialRegion(
            trial_id=i,
            region_id=region.region_id,
            active_start_index=region.active_start_index,
            active_end_index=region.active_end_index,
            threshold_start_index=region.threshold_start_index,
            threshold_end_index=region.threshold_end_index,
            cut_start_index=max(0, region.contact_start_index - pre),
            cut_end_index=min(n_rows - 1, region.active_end_index + post),
            contact_start_index=region.contact_start_index,
            contact_load_offset_N=region.contact_load_offset_N,
            contact_displacement_mm=region.contact_displacement_mm,
            displacement_excursion_mm=region.displacement_excursion_mm,
            min_valid_displacement_excursion_mm=config.min_valid_displacement_excursion_mm,
            small_displacement_load_event_flag=region.small_displacement_load_event_flag,
            valid_indentation_event_flag=region.valid_indentation_event_flag,
            excluded_reason=region.excluded_reason,
        )
        for i, region in enumerate(regions, start=1)
    ]


def zero_adjust_trial(
    trial_df: pd.DataFrame, region: TrialRegion, config: VisualizationConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trial = trial_df.copy()
    contact_start_local = int(np.argmax(trial["global_sample"].to_numpy() == region.contact_start_index))

    baseline_part = trial[trial["global_sample"] < region.contact_start_index]
    if baseline_part.empty:
        baseline_part = trial.iloc[[contact_start_local]]
    load_offset = float(baseline_part["raw_load_N"].median()) if config.zero_correction else 0.0

    contact_displacement = float(trial.iloc[contact_start_local]["raw_displacement_mm"])
    if not config.zero_correction:
        contact_displacement = 0.0

    trial["load_zeroed_N"] = trial["raw_load_N"] - load_offset
    trial["load_smooth_zeroed_N"] = trial["load_smooth_N"] - load_offset
    trial["indentation_mm"] = trial["raw_displacement_mm"] - contact_displacement

    return trial, {
        "load_offset_N": load_offset,
        "contact_start_index": region.contact_start_index,
        "contact_start_time_s": float(trial.iloc[contact_start_local]["global_time_s"]),
        "contact_displacement_mm": contact_displacement,
        "threshold_start_index": region.threshold_start_index,
        "threshold_end_index": region.threshold_end_index,
    }


def make_trial_dataframe(
    df: pd.DataFrame,
    region: TrialRegion,
    config: VisualizationConfig,
    *,
    subject: str,
    condition: str,
    source_csv: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trial = df.iloc[region.cut_start_index : region.cut_end_index + 1].copy()
    trial["trial_id"] = region.trial_id
    trial["region_id"] = region.region_id
    trial["relative_time_s"] = trial["global_time_s"] - trial["global_time_s"].iloc[0]
    trial["active_flag"] = (
        (trial["global_sample"] >= region.active_start_index)
        & (trial["global_sample"] <= region.active_end_index)
    )
    trial["threshold_active_flag"] = (
        (trial["global_sample"] >= region.threshold_start_index)
        & (trial["global_sample"] <= region.threshold_end_index)
    )
    trial, offsets = zero_adjust_trial(trial, region, config)

    max_load = float(trial.loc[trial["threshold_active_flag"], "load_smooth_detection_N"].max())
    displacement_excursion = region.displacement_excursion_mm
    small_displacement_flag = region.small_displacement_load_event_flag
    quality = "small_displacement" if small_displacement_flag else "normal"

    trial["subject"] = subject
    trial["condition"] = condition
    trial["source_csv"] = source_csv
    trial["small_displacement_load_event_flag"] = region.small_displacement_load_event_flag
    trial["valid_indentation_event_flag"] = region.valid_indentation_event_flag
    trial["excluded_reason"] = region.excluded_reason
    trial["small_displacement_flag"] = small_displacement_flag
    trial["contact_start_index"] = offsets["contact_start_index"]
    trial["contact_start_time_s"] = offsets["contact_start_time_s"]
    trial["contact_displacement_mm"] = offsets["contact_displacement_mm"]
    trial["load_offset_N"] = offsets["load_offset_N"]
    trial["cut_start_time_s"] = float(df.loc[region.cut_start_index, "global_time_s"])
    trial["cut_end_time_s"] = float(df.loc[region.cut_end_index, "global_time_s"])
    trial["threshold_start_index"] = region.threshold_start_index
    trial["threshold_end_index"] = region.threshold_end_index
    trial["threshold_end_time_s"] = float(df.loc[region.threshold_end_index, "global_time_s"])
    trial["active_end_index"] = region.active_end_index
    trial["min_valid_displacement_excursion_mm"] = region.min_valid_displacement_excursion_mm
    trial["detection_load_offset_N"] = float(df.loc[region.threshold_start_index, "detection_load_offset_N"])
    trial["event_quality_label"] = quality
    trial["displacement_excursion_mm"] = displacement_excursion
    trial["max_load_N"] = max_load
    trial["max_load_detection_N"] = max_load
    trial["threshold_start_time_s"] = float(df.loc[region.threshold_start_index, "global_time_s"])
    trial["active_end_time_s"] = float(df.loc[region.active_end_index, "global_time_s"])

    ordered = [
        "subject",
        "condition",
        "source_csv",
        "trial_id",
        "region_id",
        "global_sample",
        "raw_time_s",
        "time_s",
        "global_time_s",
        "relative_time_s",
        "raw_load_N",
        "load_N",
        "load_smooth_N",
        "load_detection_N",
        "load_smooth_detection_N",
        "load_zeroed_N",
        "load_smooth_zeroed_N",
        "raw_displacement_mm",
        "displacement_mm",
        "indentation_mm",
        "detection_load_offset_N",
        "cut_start_time_s",
        "cut_end_time_s",
        "contact_start_index",
        "contact_start_time_s",
        "contact_displacement_mm",
        "load_offset_N",
        "threshold_start_index",
        "active_flag",
        "threshold_active_flag",
        "small_displacement_load_event_flag",
        "valid_indentation_event_flag",
        "excluded_reason",
        "small_displacement_flag",
        "event_quality_label",
        "displacement_excursion_mm",
        "min_valid_displacement_excursion_mm",
        "max_load_N",
        "max_load_detection_N",
        "threshold_start_time_s",
        "threshold_end_time_s",
        "active_end_index",
        "active_end_time_s",
    ]
    contact_info = {
        **offsets,
        "subject": subject,
        "condition": condition,
        "source_csv": source_csv,
        "trial_id": region.trial_id,
        "region_id": region.region_id,
        "cut_start_time_s": float(df.loc[region.cut_start_index, "global_time_s"]),
        "cut_end_time_s": float(df.loc[region.cut_end_index, "global_time_s"]),
        "threshold_start_time_s": float(df.loc[region.threshold_start_index, "global_time_s"]),
        "threshold_end_time_s": float(df.loc[region.threshold_end_index, "global_time_s"]),
        "active_end_index": region.active_end_index,
        "active_end_time_s": float(df.loc[region.active_end_index, "global_time_s"]),
        "detection_load_offset_N": float(df.loc[region.threshold_start_index, "detection_load_offset_N"]),
        "max_load_detection_N": max_load,
        "displacement_excursion_mm": displacement_excursion,
        "max_load_N": max_load,
        "min_valid_displacement_excursion_mm": region.min_valid_displacement_excursion_mm,
        "small_displacement_load_event_flag": region.small_displacement_load_event_flag,
        "valid_indentation_event_flag": region.valid_indentation_event_flag,
        "excluded_reason": region.excluded_reason,
        "small_displacement_flag": small_displacement_flag,
        "event_quality_label": quality,
    }
    return trial[ordered], contact_info


def process_dataframe(
    raw_df: pd.DataFrame,
    config: VisualizationConfig,
    *,
    subject: str = "unknown",
    condition: str = "unknown",
    source_csv: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preprocessed = preprocess_data(raw_df, config)
    candidate_ranges = detect_load_candidate_regions(preprocessed, config)
    candidates = characterize_candidate_regions(preprocessed, candidate_ranges, config)
    accepted_candidates = [candidate for candidate in candidates if candidate.valid_indentation_event_flag]
    regions = create_trial_segments(accepted_candidates, len(preprocessed), config)
    trials: list[pd.DataFrame] = []
    contacts: list[dict[str, Any]] = []

    for region in regions:
        trial, contact = make_trial_dataframe(
            preprocessed,
            region,
            config,
            subject=subject,
            condition=condition,
            source_csv=source_csv,
        )
        trials.append(trial)
        contacts.append(contact)

    corrected = pd.concat(trials, ignore_index=True) if trials else pd.DataFrame()
    contact_points = pd.DataFrame(contacts)
    return corrected, contact_points, preprocessed


def analyze_csv(
    input_path: Path,
    output_root: Path,
    config: VisualizationConfig,
    *,
    subject: str = "unknown",
    condition: str = "unknown",
) -> Path:
    from .plotting import save_3d_trial_plots, save_overview_plot

    raw_df, encoding = read_input_csv(input_path)
    corrected, contact_points, preprocessed = process_dataframe(
        raw_df,
        config,
        subject=subject,
        condition=condition,
        source_csv=input_path.name,
    )

    output_dir = make_output_dir(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots" / "3d").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots" / "overview").mkdir(parents=True, exist_ok=True)

    corrected.to_csv(output_dir / "corrected_time_series_by_trial.csv", index=False)
    contact_points.to_csv(output_dir / "contact_points.csv", index=False)
    settings = {
        "input_csv": str(input_path),
        "csv_encoding": encoding,
        "subject": subject,
        "condition": condition,
        "config": asdict(config),
        "trial_count": int(contact_points.shape[0]),
    }
    (output_dir / "analysis_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if config.save_plots:
        save_3d_trial_plots(corrected, output_dir / "plots" / "3d")
        save_overview_plot(preprocessed, contact_points, output_dir / "plots" / "overview", input_path.stem)
    return output_dir


def make_output_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / "3d_visualization"
    candidate = base / f"batch_{timestamp}"
    if not candidate.exists():
        return candidate
    for i in range(2, 1000):
        numbered = base / f"batch_{timestamp}_{i:03d}"
        if not numbered.exists():
            return numbered
    raise RuntimeError("Could not create a unique output folder name.")
