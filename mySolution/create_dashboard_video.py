import argparse
import os
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GIF_ROOT = PROJECT_ROOT / "mySolution" / "gif_creation"
DEFAULT_ESTIMATION_ROOT = PROJECT_ROOT / "mySolution" / "results" / "saved_movement_estimations"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "mySolution" / "results" / "dashboard_videos"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTPUT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def numeric_png_files(folder):
    files = []
    for path in folder.iterdir():
        if path.suffix.lower() != ".png":
            continue
        try:
            frame_index = int(path.stem)
        except ValueError:
            continue
        files.append((frame_index, path))
    return [path for _, path in sorted(files)]


def load_estimation(video, estimation_root):
    path = estimation_root / f"video_{video}_movement_estimation.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved movement estimation: {path}")
    data = np.load(path)
    return np.asarray(data["movement_path"], dtype=float), float(data["turning_point"])


def load_measured(video):
    path = PROJECT_ROOT / "distance_labels" / f"{video}.npy"
    if not path.exists():
        return None, None
    measured = np.load(path).astype(float)
    return measured, float(np.argmax(measured))


def video_fps(video_file, fallback):
    if not video_file or not video_file.exists():
        return fallback
    capture = cv2.VideoCapture(str(video_file))
    fps = capture.get(cv2.CAP_PROP_FPS) if capture.isOpened() else 0.0
    capture.release()
    return fps if fps and fps > 0 else fallback


def letterbox(image, target_width, target_height, color=(18, 18, 18)):
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_height, target_width, 3), color, dtype=np.uint8)
    x0 = (target_width - new_width) // 2
    y0 = (target_height - new_height) // 2
    canvas[y0 : y0 + new_height, x0 : x0 + new_width] = resized
    return canvas


def setup_plot(width_px, height_px, movement_path, measured, estimated_tp, measured_tp, video):
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("#171717")
    ax.set_facecolor("#171717")

    frames = np.arange(len(movement_path))
    ax.plot(frames, movement_path, color="#1f77b4", linewidth=1.4, alpha=0.35, label="estimated full path")
    estimated_live, = ax.plot([], [], color="#1f77b4", linewidth=2.4, label="estimated path")
    estimated_point, = ax.plot([], [], "o", color="#ff4d4d", markersize=7, markeredgecolor="white", markeredgewidth=1.2)

    measured_live = None
    measured_point = None
    if measured is not None:
        measured_frames = np.arange(len(measured))
        ax.plot(measured_frames, measured, color="#eeeeee", linewidth=1.0, alpha=0.25, label="measured full path")
        measured_live, = ax.plot([], [], color="#eeeeee", linewidth=1.8, alpha=0.95, label="measured path")
        measured_point, = ax.plot([], [], "o", color="#f2f2f2", markersize=5, markeredgecolor="#222222", markeredgewidth=1.0)
        ax.axvline(measured_tp, color="#eeeeee", linestyle="--", linewidth=1.1, alpha=0.85, label="measured TP")

    ax.axvline(estimated_tp, color="#1f77b4", linestyle="--", linewidth=1.1, alpha=0.95, label="estimated TP")
    current_line = ax.axvline(0, color="#ff4d4d", linewidth=1.2, alpha=0.8)

    max_distance = float(np.nanmax(movement_path))
    if measured is not None:
        max_distance = max(max_distance, float(np.nanmax(measured)))
    ax.set_xlim(0, max(len(movement_path), len(measured) if measured is not None else 0) - 1)
    ax.set_ylim(0, max_distance * 1.08)
    ax.set_title(f"Video {video}: live nozzle position", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Frame", color="white")
    ax.set_ylabel("Nozzle position [m]", color="white")
    ax.tick_params(colors="white", labelsize=9)
    ax.grid(True, axis="y", color="white", alpha=0.16)
    for spine in ax.spines.values():
        spine.set_color("#777777")

    legend_handles = [
        plt.Line2D([0], [0], color="#1f77b4", linewidth=2.2, label="Estimated position"),
        plt.Line2D([0], [0], color="#eeeeee", linewidth=1.6, label="Measured position"),
        plt.Line2D([0], [0], color="#ff4d4d", linewidth=1.4, label="Current frame"),
        plt.Line2D([0], [0], color="#1f77b4", linestyle="--", linewidth=1.2, label="Estimated TP"),
        plt.Line2D([0], [0], color="#eeeeee", linestyle="--", linewidth=1.2, label="Measured TP"),
    ]
    legend = ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8)
    legend.get_frame().set_facecolor("#222222")
    legend.get_frame().set_edgecolor("#777777")
    for text in legend.get_texts():
        text.set_color("white")

    metric_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#222222", "edgecolor": "#555555", "alpha": 0.85},
    )

    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.22)
    return fig, ax, estimated_live, estimated_point, measured_live, measured_point, current_line, metric_text


def render_plot_frame(fig, estimated_live, estimated_point, measured_live, measured_point, current_line, metric_text, movement_path, measured, frame_index):
    path_index = min(frame_index, len(movement_path) - 1)
    x = np.arange(path_index + 1)
    estimated_live.set_data(x, movement_path[: path_index + 1])
    estimated_point.set_data([path_index], [movement_path[path_index]])
    current_line.set_xdata([frame_index, frame_index])

    lines = [f"Frame: {frame_index}", f"Estimated: {movement_path[path_index]:.2f} m"]
    if measured is not None:
        measured_index = min(frame_index, len(measured) - 1)
        measured_live.set_data(np.arange(measured_index + 1), measured[: measured_index + 1])
        measured_point.set_data([measured_index], [measured[measured_index]])
        lines.append(f"Measured: {measured[measured_index]:.2f} m")
        lines.append(f"Abs. error: {abs(movement_path[path_index] - measured[measured_index]):.2f} m")

    metric_text.set_text("\n".join(lines))
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def create_dashboard_video(video, frame_folder, estimation_root, output_path, source_video, fps, width, height):
    frame_files = numeric_png_files(frame_folder)
    if not frame_files:
        raise FileNotFoundError(f"No numbered PNG frames found in {frame_folder}")

    movement_path, estimated_tp = load_estimation(video, estimation_root)
    measured, measured_tp = load_measured(video)
    fps = video_fps(source_video, fps)

    left_width = width // 2
    right_width = width - left_width
    fig, ax, estimated_live, estimated_point, measured_live, measured_point, current_line, metric_text = setup_plot(
        right_width,
        height,
        movement_path,
        measured,
        estimated_tp,
        measured_tp,
        video,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for index, frame_path in enumerate(frame_files):
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read frame {frame_path}")

            image_panel = letterbox(image, left_width, height)
            plot_panel = render_plot_frame(
                fig,
                estimated_live,
                estimated_point,
                measured_live,
                measured_point,
                current_line,
                metric_text,
                movement_path,
                measured,
                index,
            )
            plot_panel = cv2.resize(plot_panel, (right_width, height), interpolation=cv2.INTER_AREA)

            cv2.putText(image_panel, f"Frame {index}", (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(
                image_panel,
                f"Estimated distance: {movement_path[min(index, len(movement_path) - 1)]:.2f} m",
                (28, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (80, 230, 80),
                2,
            )

            writer.write(np.hstack([image_panel, plot_panel]))
            if index > 0 and index % 100 == 0:
                print(f"Rendered {index}/{len(frame_files)} frames")
    finally:
        writer.release()
        plt.close(fig)

    print(f"Saved dashboard video to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create side-by-side sewer video and live movement-path dashboard.")
    parser.add_argument("--video", type=int, default=11)
    parser.add_argument("--frame-folder", type=Path, default=None)
    parser.add_argument("--source-video", type=Path, default=None)
    parser.add_argument("--estimation-root", type=Path, default=DEFAULT_ESTIMATION_ROOT)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main():
    args = parse_args()
    frame_folder = args.frame_folder or DEFAULT_GIF_ROOT / str(args.video)
    source_video = args.source_video
    if source_video is None:
        mp4_files = sorted(DEFAULT_GIF_ROOT.glob("*.mp4"))
        source_video = mp4_files[0] if mp4_files else None
    output_path = args.output_path or DEFAULT_OUTPUT_ROOT / f"video_{args.video}_dashboard.mp4"

    create_dashboard_video(
        args.video,
        frame_folder,
        args.estimation_root,
        output_path,
        source_video,
        args.fps,
        args.width,
        args.height,
    )


if __name__ == "__main__":
    main()
