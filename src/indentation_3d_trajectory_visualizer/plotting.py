from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_3d_trial_plots(corrected: pd.DataFrame, output_dir: Path) -> None:
    if corrected.empty:
        return
    for trial_id, trial in corrected.groupby("trial_id"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        active = trial[trial["active_flag"]]
        inactive = trial[~trial["active_flag"]]

        if not inactive.empty:
            ax.plot(
                inactive["relative_time_s"],
                inactive["indentation_mm"],
                inactive["load_zeroed_N"],
                color="#9ca3af",
                linewidth=1.0,
                alpha=0.55,
                label="margin",
            )
        ax.plot(
            active["relative_time_s"],
            active["indentation_mm"],
            active["load_zeroed_N"],
            color="#2563eb",
            linewidth=1.8,
            label="active",
        )
        contact = trial.iloc[(trial["global_sample"] - trial["contact_start_index"].iloc[0]).abs().argmin()]
        ax.scatter(
            [contact["relative_time_s"]],
            [contact["indentation_mm"]],
            [contact["load_zeroed_N"]],
            color="#dc2626",
            s=36,
            label="contact start",
        )
        label = str(trial["event_quality_label"].iloc[0])
        title = f"{trial['subject'].iloc[0]} {trial['condition'].iloc[0]} trial {trial_id} ({label})"
        ax.set_title(title)
        ax.set_xlabel("relative time (s)")
        ax.set_ylabel("indentation (mm)")
        ax.set_zlabel("load zeroed (N)")
        ax.legend(loc="upper left")
        fig.tight_layout()

        stem = f"{_safe_name(trial['subject'].iloc[0])}_{_safe_name(trial['condition'].iloc[0])}_trial_{int(trial_id):02d}_3d.png"
        fig.savefig(output_dir / stem, dpi=180)
        plt.close(fig)


def save_overview_plot(
    preprocessed: pd.DataFrame,
    contact_points: pd.DataFrame,
    output_dir: Path,
    stem: str,
) -> None:
    if preprocessed.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(preprocessed["global_time_s"], preprocessed["raw_load_N"], color="#64748b", linewidth=1, label="raw")
    axes[0].plot(
        preprocessed["global_time_s"],
        preprocessed["load_smooth_N"],
        color="#2563eb",
        linewidth=1.2,
        label="smooth",
    )
    axes[0].set_ylabel("load (N)")
    axes[0].legend(loc="upper right")

    axes[1].plot(
        preprocessed["global_time_s"],
        preprocessed["raw_displacement_mm"],
        color="#16a34a",
        linewidth=1.0,
    )
    axes[1].set_xlabel("global time (s)")
    axes[1].set_ylabel("displacement (mm)")

    if not contact_points.empty:
        for _, row in contact_points.iterrows():
            for ax in axes:
                ax.axvline(row["contact_start_time_s"], color="#dc2626", linewidth=0.9, alpha=0.7)

    fig.suptitle("Raw signal and contact starts")
    fig.tight_layout()
    fig.savefig(output_dir / f"{_safe_name(stem)}_raw_contact_overview.png", dpi=180)
    plt.close(fig)


def _safe_name(value: object) -> str:
    text = str(value).strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text) or "unknown"

