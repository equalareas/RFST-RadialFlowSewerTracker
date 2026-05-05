import argparse
import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MY_SOLUTION_ROOT = PROJECT_ROOT / "mySolution"
DEFAULT_FRAME_ROOT = PROJECT_ROOT / "bonus_frame_images"
DEFAULT_HEURISTIC_ROOT = MY_SOLUTION_ROOT / "results" / "heuristic_bonus_detector"
DEFAULT_ESTIMATION_ROOT = MY_SOLUTION_ROOT / "results" / "saved_bonus_movement_estimations"
OUTPUT_ROOT = MY_SOLUTION_ROOT / "results" / "bonus_inspection_overlays"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


INTERVAL_STYLES = {
    "no_vision": {"color": "#d62728", "alpha": 0.18, "name": "No vision"},
    "dirt_or_deposit": {"color": "#8c6d31", "alpha": 0.16, "name": "Dirt/deposit"},
    "anomaly_check": {"color": "#9467bd", "alpha": 0.22, "name": "Anomaly"},
}
INTERVAL_LABEL_Y_FRACTIONS = {
    "anomaly_check": 0.94,
    "dirt_or_deposit": 0.82,
}
MANUAL_TURNING_POINTS = {
    1: 10852,
    2: 1770,
    3: 875,
    4: 2377,
    5: 2325,
    6: 1524,
    7: 731,
    8: 394,
    9: 4252,
}

MAIN_LEGEND_LOC = "lower left"
MAIN_LEGEND_BBOX = (-0.01, -0.29)
MAIN_LEGEND_COLUMNS = 2
MAIN_LEGEND_ROWS = 3
METRICS_BOX_XY = (0.32, -0.15)
VISIBILITY_LEGEND_LOC = "lower right"
VISIBILITY_LEGEND_BBOX = (1, -2.5)
FIGURE_BOTTOM_MARGIN = 0.20

VISIBILITY_CMAP = ListedColormap(["#ffffff", "#d62728", "#f0ad4e", "#2ca02c"])


def fixed_grid_legend(ax, handles, labels, loc, bbox_to_anchor, ncol, nrows, **kwargs):
    visible = [(handle, label) for handle, label in zip(handles, labels) if label]
    target_count = ncol * nrows
    if len(visible) > target_count:
        nrows = int(np.ceil(len(visible) / ncol))
        target_count = ncol * nrows

    spacer_count = target_count - len(visible)
    for index in range(spacer_count):
        visible.append((plt.Line2D([], [], linestyle="", alpha=0.0), " "))

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


def available_videos(frame_root):
    return sorted(int(path.name) for path in frame_root.iterdir() if path.is_dir() and path.name.isdigit())


def frame_count(video, frame_root):
    folder = frame_root / str(video)
    return sum(1 for path in folder.iterdir() if path.suffix.lower() == ".png")


def heuristic_inspection_bounds(intervals):
    for interval in intervals:
        if interval["label"] == "inspection_active":
            return int(float(interval["start_frame"])), int(float(interval["end_frame"]))
    return None


def load_saved_estimation(video, estimation_root):
    path = estimation_root / f"bonus_video_{video}_movement_estimation.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "video": int(data["video"]),
        "movement_path": np.asarray(data["movement_path"], dtype=float),
        "movement_direction": np.asarray(data["movement_direction"], dtype=float),
        "turning_point": float(data["turning_point"]),
        "channel_length": float(data["channel_length"]) if "channel_length" in data else float(np.max(data["movement_path"])),
    }


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


def choose_inspection_bounds(intervals, estimated):
    if estimated is not None:
        bounds = movement_inspection_bounds(estimated["movement_path"])
        if bounds is not None:
            return bounds
    return heuristic_inspection_bounds(intervals)


def mark_inspection_bounds(ax, bounds, y_top):
    if bounds is None:
        return
    start, end = bounds
    ax.axvline(start, color="#17becf", linestyle="-", linewidth=1.5, label="Inspection start")
    ax.axvline(end, color="#17becf", linestyle=":", linewidth=1.7, label="Inspection end")
    ax.annotate("inspection start", xy=(start, y_top * 0.96), xytext=(6, 0), textcoords="offset points",
                ha="left", va="top", fontsize=8, color="#0f6670")
    ax.annotate("inspection end", xy=(end, y_top * 0.96), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="top", fontsize=8, color="#0f6670")


def shade_intervals(ax, intervals, y_top, bounds=None):
    for interval in intervals:
        label = interval["label"]
        if label in {"pre_inspection_handling", "inspection_active", "post_inspection_handling"}:
            continue
        style = INTERVAL_STYLES.get(label, {"color": "#7f7f7f", "alpha": 0.15, "name": label})
        start = int(float(interval["start_frame"]))
        end = int(float(interval["end_frame"]))
        if bounds is not None:
            start = max(start, bounds[0])
            end = min(end, bounds[1])
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


def visibility_stage(score):
    if score >= 0.93:
        return 2
    if score >= 0.80:
        return 1
    return 0


def plot_visibility_bar(ax, scores, bounds):
    if not scores:
        ax.text(0.5, 0.5, "No visibility scores found", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    frames = np.array([int(row["frame"]) for row in scores], dtype=np.int64)
    visibility = np.array([float(row["visibility_score_smooth"]) for row in scores], dtype=np.float32)
    stages = np.array([visibility_stage(value) for value in visibility], dtype=np.int64) + 1
    if bounds is not None:
        stages = stages.copy()
        stages[(frames < bounds[0]) | (frames > bounds[1])] = 0
    ax.imshow(
        stages.reshape(1, -1),
        aspect="auto",
        cmap=VISIBILITY_CMAP,
        vmin=0,
        vmax=3,
        extent=[frames[0], frames[-1], 0, 1],
        interpolation="nearest",
    )
    ax.set_yticks([])
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


def format_metric_box(estimated, manual_turning_point):
    if estimated is None:
        return "No movement estimate"
    if manual_turning_point is None:
        return "Manual TP unavailable"
    tp_error = abs(float(estimated["turning_point"]) - float(manual_turning_point))
    return f"TP error: {tp_error:.0f} frames"


def add_metric_box(ax, estimated, manual_turning_point):
    ax.text(
        METRICS_BOX_XY[0],
        METRICS_BOX_XY[1],
        format_metric_box(estimated, manual_turning_point),
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


def movement_legend_items(has_manual_turning_point):
    handles = [
        plt.Line2D([0], [0], color="#17becf", linestyle="-", linewidth=1.5),
        plt.Line2D([0], [0], color="#17becf", linestyle=":", linewidth=1.7),
        plt.Line2D([0], [0], color="#1f77b4", linewidth=1.5),
        plt.Line2D([0], [0], color="#111111", linewidth=1.0, alpha=0.25),
        plt.Line2D([0], [0], color="#1f77b4", linestyle="--", linewidth=1.0),
        plt.Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.0),
    ]
    labels = [
        "Inspection start",
        "Inspection end",
        "Estimated position",
        "Measured position (n/a)",
        "Estimated turning point",
        "Manual turning point",
    ]
    if not has_manual_turning_point:
        handles[5].set_alpha(0.25)
        labels[5] = "Manual turning point (n/a)"
    return handles, labels


def plot_video(video, frame_root, heuristic_root, estimation_root):
    intervals = read_csv(heuristic_root / "intervals" / f"video_{video}_intervals.csv")
    scores = read_csv(heuristic_root / "scores" / f"video_{video}_scores.csv")
    estimated = load_saved_estimation(video, estimation_root)
    manual_turning_point = MANUAL_TURNING_POINTS.get(video)
    n_frames = frame_count(video, frame_root)
    bounds = choose_inspection_bounds(intervals, estimated)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [5.2, 0.28], "hspace": 0.04},
    )

    if estimated is not None:
        movement_path = estimated["movement_path"]
        y_top = max(float(np.max(movement_path)), 1.0)
        x = np.arange(len(movement_path))
        shade_intervals(axes[0], intervals, y_top, bounds=bounds)
        mark_inspection_bounds(axes[0], bounds, y_top)
        axes[0].plot(x, movement_path, color="#1f77b4", linewidth=1.5, label="Estimated position")
        axes[0].axvline(estimated["turning_point"], color="#1f77b4", linestyle="--", linewidth=1.0)
        if manual_turning_point is not None:
            axes[0].axvline(manual_turning_point, color="#111111", linestyle="--", linewidth=1.0)
        axes[0].set_ylim(0, y_top * 1.05)
        axes[0].set_ylabel("Nozzle position [m]")
        add_metric_box(axes[0], estimated, manual_turning_point)
    else:
        y_top = 1.0
        shade_intervals(axes[0], intervals, y_top, bounds=bounds)
        mark_inspection_bounds(axes[0], bounds, y_top)
        axes[0].plot([0, max(0, n_frames - 1)], [0.5, 0.5], color="#1f77b4", linewidth=1.2)
        axes[0].set_ylim(0, 1)
        axes[0].set_yticks([])
        add_metric_box(axes[0], estimated, manual_turning_point)

    axes[0].set_title(f"Bonus video {video}: estimated nozzle position with inspection overlays")
    axes[0].tick_params(labelbottom=False)
    axes[0].grid(True, axis="y", alpha=0.25)

    plot_visibility_bar(axes[1], scores, bounds)

    legend_handles, legend_labels = movement_legend_items(manual_turning_point is not None)
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

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=FIGURE_BOTTOM_MARGIN, hspace=0.04)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"bonus_video_{video}_inspection_overlay.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot heuristic inspection overlays for bonus videos.")
    parser.add_argument("--video", type=int, default=None)
    parser.add_argument("--frame-root", type=Path, default=DEFAULT_FRAME_ROOT)
    parser.add_argument("--heuristic-root", type=Path, default=DEFAULT_HEURISTIC_ROOT)
    parser.add_argument("--estimation-root", type=Path, default=DEFAULT_ESTIMATION_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    videos = [args.video] if args.video is not None else available_videos(args.frame_root)
    for video in videos:
        output_path = plot_video(video, args.frame_root, args.heuristic_root, args.estimation_root)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
