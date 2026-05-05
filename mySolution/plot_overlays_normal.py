import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MY_SOLUTION_ROOT = PROJECT_ROOT / "mySolution"
DEFAULT_HEURISTIC_ROOT = MY_SOLUTION_ROOT / "results" / "heuristic_main_detector"
DEFAULT_ESTIMATION_ROOT = MY_SOLUTION_ROOT / "results" / "saved_movement_estimations"
DEFAULT_EVALUATION_PATH = DEFAULT_ESTIMATION_ROOT / "evaluation.csv"
OUTPUT_ROOT = MY_SOLUTION_ROOT / "results" / "movement_inspection_overlays"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

sys.path.insert(0, str(MY_SOLUTION_ROOT))
from MovementPath import MovementPath
from MovementPathEstimator import MovementPathEstimator


INTERVAL_STYLES = {
    "no_vision": {"color": "#d62728", "alpha": 0.18, "name": "No vision"},
    "dirt_or_deposit": {"color": "#8c6d31", "alpha": 0.16, "name": "Dirt/deposit"},
    "anomaly_check": {"color": "#9467bd", "alpha": 0.22, "name": "Anomaly"},
    "roots": {"color": "#2ca02c", "alpha": 0.18, "name": "Roots"},
}
INTERVAL_LABEL_Y_FRACTIONS = {
    "anomaly_check": 0.94,
    "dirt_or_deposit": 0.82,
}

# Manual layout knobs, in axes coordinates.
# Use these when the final figure needs small presentation tweaks.
MAIN_LEGEND_LOC = "lower left"
MAIN_LEGEND_BBOX = (-0.01, -0.29)
MAIN_LEGEND_COLUMNS = 2
MAIN_LEGEND_ROWS = 3
METRICS_BOX_XY = (0.3, -0.15)
METRICS_BOX_XY_NO_GROUND_TRUTH = (0.32, -0.15)
VISIBILITY_LEGEND_LOC = "lower right"
VISIBILITY_LEGEND_BBOX = (1, -2.5)
FIGURE_BOTTOM_MARGIN = 0.2


def fixed_grid_legend(ax, handles, labels, loc, bbox_to_anchor, ncol, nrows, **kwargs):
    visible = [(handle, label) for handle, label in zip(handles, labels) if label]
    target_count = ncol * nrows
    if len(visible) > target_count:
        nrows = int(np.ceil(len(visible) / ncol))
        target_count = ncol * nrows

    spacer_count = target_count - len(visible)
    for index in range(spacer_count):
        visible.append((plt.Line2D([], [], linestyle="", alpha=0.0), f" "))

    final_handles = [item[0] for item in visible]
    final_labels = [item[1] for item in visible]
    legend_kwargs = {"loc": loc, "ncol": ncol, **kwargs}
    if bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = bbox_to_anchor
    legend = ax.legend(final_handles, final_labels, **legend_kwargs)

    for handle, text in zip(legend.legend_handles, legend.get_texts()):
        if not text.get_text().strip():
            handle.set_visible(False)
            text.set_visible(False)
    return legend


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_evaluation_rows(path):
    rows = {}
    for row in read_csv(path):
        try:
            video = int(row["video"])
        except (ValueError, TypeError):
            continue
        rows[video] = row
    return rows


def available_videos(frame_root):
    return sorted(int(path.name) for path in frame_root.iterdir() if path.is_dir() and path.name.isdigit())


def estimate_movement(video):
    estimator = MovementPathEstimator(video, False)
    estimator.execute_estimations()
    return estimator.calculated_movement_paths[video]


def load_saved_estimation(video, estimation_root):
    path = estimation_root / f"video_{video}_movement_estimation.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return MovementPath(
        int(data["video"]),
        data["movement_path"],
        data["movement_direction"],
        float(data["turning_point"]),
    )


def load_measured(video):
    label_path = PROJECT_ROOT / "distance_labels" / f"{video}.npy"
    if not label_path.exists():
        return None
    return MovementPath(video, np.load(label_path))


def shade_intervals(ax, intervals, y_top, bounds=None):
    for interval in intervals:
        label = interval["label"]
        if label in {"pre_inspection_handling", "inspection_active", "post_inspection_handling"}:
            continue
        style = INTERVAL_STYLES.get(label, {"color": "#7f7f7f", "alpha": 0.15, "name": label})
        start = int(float(interval["start_frame"]))
        end = int(float(interval["end_frame"]))
        if bounds is not None:
            inspection_start, inspection_end = bounds
            start = max(start, inspection_start)
            end = min(end, inspection_end)
            if end < start:
                continue
        ax.axvspan(start, end, color=style["color"], alpha=style["alpha"], linewidth=0)

        if label in {"anomaly_check", "dirt_or_deposit"}:
            label_text = "anomaly" if label == "anomaly_check" else "dirt/deposit"
            label_color = "#4b2778" if label == "anomaly_check" else "#6f4e1e"
            edge_color = "#9467bd" if label == "anomaly_check" else "#8c6d31"
            ax.annotate(
                f"{label_text}\n{start}-{end}",
                xy=(start + (end - start) / 2, y_top * INTERVAL_LABEL_Y_FRACTIONS[label]),
                ha="center",
                va="top",
                fontsize=8,
                color=label_color,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": edge_color,
                    "alpha": 0.9,
                },
            )


def heuristic_inspection_bounds(intervals):
    for interval in intervals:
        if interval["label"] == "inspection_active":
            return int(float(interval["start_frame"])), int(float(interval["end_frame"]))
    return None


def movement_inspection_bounds(movement_path):
    path = np.asarray(movement_path, dtype=float)
    if path.size == 0:
        return None
    max_distance = float(np.nanmax(path))
    if max_distance <= 0:
        return None

    threshold = max(0.5, 0.01 * max_distance)
    active = path > threshold
    if not np.any(active):
        threshold = max(0.1, 0.003 * max_distance)
        active = path > threshold
    if not np.any(active):
        return None

    start = int(np.argmax(active))
    end = int(path.size - 1 - np.argmax(active[::-1]))
    return start, end


def choose_inspection_bounds(intervals, measured, estimated):
    if measured is not None:
        bounds = movement_inspection_bounds(measured.movement_path)
        if bounds is not None:
            return bounds, "measured_position"

    bounds = movement_inspection_bounds(estimated.movement_path)
    if bounds is not None:
        return bounds, "estimated_position"

    bounds = heuristic_inspection_bounds(intervals)
    if bounds is not None:
        return bounds, "heuristic_visual"
    return None, "none"


def mark_inspection_bounds(ax, bounds, y_top):
    if bounds is None:
        return
    start, end = bounds
    ax.axvline(start, color="#17becf", linestyle="-", linewidth=1.5, label="Inspection start")
    ax.axvline(end, color="#17becf", linestyle=":", linewidth=1.7, label="Inspection end")
    ax.annotate(
        "inspection start",
        xy=(start, y_top * 0.96),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color="#0f6670",
    )
    ax.annotate(
        "inspection end",
        xy=(end, y_top * 0.96),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color="#0f6670",
    )


VISIBILITY_CMAP = ListedColormap(["#ffffff", "#d62728", "#f0ad4e", "#2ca02c"])


def visibility_stage(score):
    if score >= 0.93:
        return 2
    if score >= 0.80:
        return 1
    return 0


def plot_visibility_bar(ax, scores, intervals, bounds):
    if not scores:
        ax.text(0.5, 0.5, "No visibility scores found", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    frames = np.array([int(row["frame"]) for row in scores], dtype=np.int64)
    visibility = np.array([float(row["visibility_score_smooth"]) for row in scores], dtype=np.float32)
    stages = np.array([visibility_stage(value) for value in visibility], dtype=np.int64)
    stages = stages + 1

    if bounds is not None:
        inspection_start, inspection_end = bounds
        stages = stages.copy()
        stages[(frames < inspection_start) | (frames > inspection_end)] = 0

    bar = stages.reshape(1, -1)
    extent = [frames[0], frames[-1], 0, 1]
    ax.imshow(bar, aspect="auto", cmap=VISIBILITY_CMAP, vmin=0, vmax=3, extent=extent, interpolation="nearest")
    ax.set_yticks([])
    # ax.set_ylabel("Visibility", rotation=0, ha="right", va="center", labelpad=36)
    ax.set_xlabel("Frame")

    legend_handles = [
        plt.Line2D([0], [0], color="#2ca02c", linewidth=6, label="Good visibility"),
        plt.Line2D([0], [0], color="#f0ad4e", linewidth=6, label="Reduced visibility"),
        plt.Line2D([0], [0], color="#d62728", linewidth=6, label="Poor/no visibility"),
    ]
    ax.legend(
    handles=legend_handles,
    loc=VISIBILITY_LEGEND_LOC,
    bbox_to_anchor=VISIBILITY_LEGEND_BBOX,
    fontsize=8,
    ncol=3,
    framealpha=0.92,
)


def format_metric_box(evaluation_row):
    if evaluation_row is None or evaluation_row.get("has_ground_truth") != "True":
        return "No ground truth\nError metrics unavailable"

    mae = float(evaluation_row["path_mae_m"])
    norm_mae = float(evaluation_row["path_mae_normalized"])
    tp_error = float(evaluation_row["turning_point_error_frames"])
    direction_accuracy = float(evaluation_row["direction_accuracy"])
    return (
        f"MAE: {mae:.2f} m ({100 * norm_mae:.1f}%)\n"
        f"TP error: {tp_error:.0f} frames\n"
        f"Dir acc: {100 * direction_accuracy:.1f}%"
    )


def add_metric_box(ax, evaluation_row):
    has_ground_truth = evaluation_row is not None and evaluation_row.get("has_ground_truth") == "True"
    metrics_box_xy = METRICS_BOX_XY if has_ground_truth else METRICS_BOX_XY_NO_GROUND_TRUTH
    ax.text(
        metrics_box_xy[0],
        metrics_box_xy[1],
        format_metric_box(evaluation_row),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#bdbdbd",
            "alpha": 0.9,
        },
    )


def movement_legend_items(has_measured):
    handles = [
        plt.Line2D([0], [0], color="#17becf", linestyle="-", linewidth=1.5),
        plt.Line2D([0], [0], color="#17becf", linestyle=":", linewidth=1.7),
        plt.Line2D([0], [0], color="#1f77b4", linewidth=1.5),
        plt.Line2D([0], [0], color="#111111", linewidth=1.0, alpha=0.75),
        plt.Line2D([0], [0], color="#1f77b4", linestyle="--", linewidth=1.0),
        plt.Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.0),
    ]
    labels = [
        "Inspection start",
        "Inspection end",
        "Estimated position",
        "Measured position",
        "Estimated turning point",
        "Measured turning point",
    ]
    if not has_measured:
        handles[3].set_alpha(0.25)
        handles[5].set_alpha(0.25)
        labels[3] = "Measured position (n/a)"
        labels[5] = "Measured turning point (n/a)"
    return handles, labels


def plot_video(video, heuristic_root, estimation_root, evaluation_rows):
    estimated = load_saved_estimation(video, estimation_root)
    if estimated is None:
        estimated = estimate_movement(video)
    measured = load_measured(video)
    intervals = read_csv(heuristic_root / "intervals" / f"video_{video}_intervals.csv")
    scores = read_csv(heuristic_root / "scores" / f"video_{video}_scores.csv")
    evaluation_row = evaluation_rows.get(video)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [5.2, 0.28], "hspace": 0.04},
    )

    max_distance = float(max(np.max(estimated.movement_path), np.max(measured.movement_path) if measured else 0.0))
    bounds, bounds_source = choose_inspection_bounds(intervals, measured, estimated)
    shade_intervals(axes[0], intervals, max_distance, bounds=bounds)
    mark_inspection_bounds(axes[0], bounds, max_distance)
    axes[0].plot(estimated.movement_path, color="#1f77b4", linewidth=1.5, label="Estimated position")
    if measured is not None:
        axes[0].plot(measured.movement_path, color="#111111", linewidth=1.0, alpha=0.75, label="Measured position")
        axes[0].axvline(measured.turning_point, color="#111111", linestyle="--", linewidth=1.0, label="Measured turning point")
    axes[0].axvline(estimated.turning_point, color="#1f77b4", linestyle="--", linewidth=1.0, label="Estimated turning point")
    axes[0].set_ylabel("Nozzle position [m]")
    axes[0].set_title(f"Video {video}: nozzle position with inspection overlays")
    axes[0].tick_params(labelbottom=False)
    axes[0].grid(True, axis="y", alpha=0.25)
    add_metric_box(axes[0], evaluation_row)

    plot_visibility_bar(axes[1], scores, intervals, bounds)

    legend_handles, legend_labels = movement_legend_items(measured is not None)
    fixed_grid_legend(
        axes[0],
        legend_handles,
        legend_labels,
        loc=MAIN_LEGEND_LOC,
        bbox_to_anchor=MAIN_LEGEND_BBOX,
        ncol=MAIN_LEGEND_COLUMNS,
        nrows=MAIN_LEGEND_ROWS,
        fontsize=8,
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=FIGURE_BOTTOM_MARGIN, hspace=0.04)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"video_{video}_movement_inspection_overlay.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path, bounds, bounds_source


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay inspection intervals on nozzle position plots.")
    parser.add_argument("--video", type=int, default=None, help="Video number to plot. Defaults to all frame_images videos.")
    parser.add_argument(
        "--heuristic-root",
        type=Path,
        default=DEFAULT_HEURISTIC_ROOT,
        help="Heuristic detector output folder for the same frame set.",
    )
    parser.add_argument(
        "--estimation-root",
        type=Path,
        default=DEFAULT_ESTIMATION_ROOT,
        help="Folder containing saved movement estimation .npz files.",
    )
    parser.add_argument(
        "--evaluation-path",
        type=Path,
        default=DEFAULT_EVALUATION_PATH,
        help="CSV file with saved-estimation evaluation metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame_root = PROJECT_ROOT / "frame_images"
    videos = [args.video] if args.video is not None else available_videos(frame_root)
    evaluation_rows = load_evaluation_rows(args.evaluation_path)
    summary_rows = []
    for video in videos:
        output_path, bounds, bounds_source = plot_video(
            video,
            args.heuristic_root,
            args.estimation_root,
            evaluation_rows,
        )
        if bounds is not None:
            summary_rows.append(
                {
                    "video": video,
                    "inspection_start_frame": bounds[0],
                    "inspection_end_frame": bounds[1],
                    "source": bounds_source,
                }
            )
        print(f"Saved {output_path}")

    if summary_rows:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        summary_path = OUTPUT_ROOT / "inspection_bounds_summary.csv"
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["video", "inspection_start_frame", "inspection_end_frame", "source"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
