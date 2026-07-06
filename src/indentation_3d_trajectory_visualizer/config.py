from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class VisualizationConfig:
    sample_interval_sec: float = 0.0005
    time_column: str | int | None = None
    load_column: str | int | None = None
    displacement_column: str | int | None = None
    load_threshold_n: float = 0.05
    smoothing_window_samples: int = 21
    pre_margin_sec: float = 2.0
    post_margin_sec: float = 2.0
    min_active_duration_sec: float = 0.2
    max_gap_within_trial_sec: float = 0.1
    zero_correction: bool = True
    min_valid_displacement_excursion_mm: float = 0.05
    save_trial_csv: bool = True
    save_plots: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
