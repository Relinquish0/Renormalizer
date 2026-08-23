#!/usr/bin/env python3
"""Select RMSD thresholds and draw the temperature--failure-time diagram."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from threshold_analysis import (
    Comparison,
    TEMPERATURES,
    convergence_audit,
    interval_error_audit,
    threshold_sweep,
)


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent

# Selected after auditing several chi ladders and epsilon values.
FAIL_EPSILON = 1e-2
CONVERGENCE_EPSILON = 1e-3
SENSITIVITY_EPSILONS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
PLOT_Y_MAX = 8.0
ABOVE_LIMIT_MARKER_Y = 7.88
SELECTED_COMPARISONS = (
    Comparison("singleset", 16, 32),
    Comparison("multiset", 16, 32),
)


STYLE = {
    "singleset": {"color": "#C65D3B", "marker": "o", "label": "Singleset $\\chi=16$"},
    "multiset": {"color": "#247BA0", "marker": "s", "label": "Multiset $\\chi=16$"},
}


def select_rows(sweep: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    keep = np.isclose(sweep["epsilon"], epsilon)
    keep &= sweep["chi"].eq(16)
    keep &= sweep["reference_chi"].eq(32)
    keep &= sweep["reference_method"].eq("multiset")
    return sweep.loc[keep].sort_values(["method", "temperature"]).copy()


def plot_decision_map(selected: pd.DataFrame, output_stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.1), constrained_layout=True)

    curves = {}
    for method in ("singleset", "multiset"):
        frame = selected[selected["method"].eq(method)].sort_values("temperature")
        x = frame["temperature"].to_numpy()
        y = frame["t_epsilon"].to_numpy()
        y_display = np.where(y > PLOT_Y_MAX, ABOVE_LIMIT_MARKER_Y, y)
        curves[method] = (x, y_display)
        style = STYLE[method]
        ax.plot(
            x,
            y_display,
            lw=2.2,
            ms=6.2,
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            zorder=3,
        )
        above_limit = y > PLOT_Y_MAX
        if np.any(above_limit):
            ax.scatter(
                x[above_limit],
                np.full(np.count_nonzero(above_limit), ABOVE_LIMIT_MARKER_Y),
                marker="^",
                s=78,
                facecolors="white",
                edgecolors=style["color"],
                linewidths=1.8,
                clip_on=True,
                zorder=5,
            )
            for x_value, y_value in zip(x[above_limit], y[above_limit]):
                ax.annotate(
                    rf"$>{PLOT_Y_MAX:g}$ ({y_value:.2f})",
                    (x_value, ABOVE_LIMIT_MARKER_Y),
                    xytext=(5, -12),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=style["color"],
                    ha="left",
                    va="top",
                )
        censored = frame["censored"].to_numpy(dtype=bool)
        if np.any(censored):
            ax.scatter(
                x[censored], y[censored], marker="^", s=65,
                facecolors="none", edgecolors=style["color"], zorder=4,
            )

    x_single, y_single = curves["singleset"]
    x_multi, y_multi = curves["multiset"]
    if not np.allclose(x_single, x_multi):
        raise ValueError("method curves do not share the same temperature grid")
    ax.fill_between(
        x_single,
        y_single,
        y_multi,
        where=y_multi >= y_single,
        interpolate=True,
        color="#8AC6A8",
        alpha=0.34,
        label="Multiset advantage window",
        zorder=1,
    )

    ax.set_xlabel(r"Reduced temperature $T^*=k_\mathrm{B}T/\omega_0$")
    ax.set_ylabel(r"Failure time $t_{\epsilon}/(2\pi)$")
    ax.set_title(
        r"RMSD convergence boundary at $\chi=16$"
        "\n"
        rf"Common reference: multiset $\chi=32$; "
        rf"$\epsilon_{{\rm fail}}={FAIL_EPSILON:g}$"
    )
    ax.set_xticks(TEMPERATURES)
    ax.set_xlim(-0.08, 4.08)
    ax.set_ylim(0, PLOT_Y_MAX)
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(frameon=True, framealpha=0.96, loc="upper right")

    note = (
        r"Above each boundary: RMSD error $>0.01$." "\n"
        r"Shading: only multiset $\chi=16$ remains accurate."
    )
    ax.text(
        0.52,
        0.46,
        note,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.6,
        color="0.28",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.82", "alpha": 0.92},
    )

    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir", type=Path, default=BASE_DIR,
        help="directory containing the Holstein77_T* result folders",
    )
    parser.add_argument(
        "--output-stem", type=Path, default=HERE / "fig3_tepsilon_temperature",
        help="output path without extension",
    )
    args = parser.parse_args()

    # The full table supplies the requested chi/epsilon sensitivity analysis.
    sweep = threshold_sweep(
        args.base_dir,
        comparisons=(
            Comparison("singleset", 16, 32),
            Comparison("singleset", 32, 32),
            Comparison("singleset", 64, 32),
            Comparison("multiset", 8, 32),
            Comparison("multiset", 16, 32),
        ),
        epsilons=SENSITIVITY_EPSILONS,
    )
    sweep.to_csv(HERE / "threshold_sensitivity.csv", index=False)

    audit = convergence_audit(
        args.base_dir,
        fail_epsilon=FAIL_EPSILON,
        convergence_epsilon=CONVERGENCE_EPSILON,
    )
    audit.to_csv(HERE / "threshold_selection_audit.csv", index=False)
    if not bool(audit["multiset_converged"].all()):
        failed = audit.loc[~audit["multiset_converged"], "temperature"].tolist()
        raise RuntimeError(f"selected thresholds fail the method-separation audit at T*={failed}")

    selected = select_rows(sweep, FAIL_EPSILON)
    selected.to_csv(HERE / "selected_tepsilon.csv", index=False)

    interval_audit = interval_error_audit(
        args.base_dir,
        comparison=Comparison("multiset", 16, 32),
        epsilon=FAIL_EPSILON,
        t_start=0.0,
        t_stop=8.0,
    )
    interval_audit.to_csv(
        HERE / "multiset_chi16_vs32_t0_8_audit.csv", index=False
    )
    plot_decision_map(selected, args.output_stem)

    print("Selected thresholds:")
    print(f"  epsilon_conv = {CONVERGENCE_EPSILON:g}")
    print(f"  epsilon_fail = {FAIL_EPSILON:g}")
    print("\nCommon-reference method-separation audit:")
    print(audit.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\nSelected t_epsilon values:")
    print(
        selected[
            [
                "temperature", "method", "chi", "reference_method",
                "reference_chi", "t_epsilon", "censored",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
    )
    print("\nMultiset chi=16 vs 32 audit on t=0--8 a.u.:")
    print(
        interval_audit[
            [
                "temperature", "max_error", "time_of_max_error",
                "first_crossing", "within_tolerance",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
    )
    print(f"\nWrote {args.output_stem.with_suffix('.png')}")
    print(f"Wrote {args.output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
