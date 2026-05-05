import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME_ROOT = PROJECT_ROOT / "bonus_frame_images"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "mySolution" / "results" / "heuristic_bonus_detector"


def numeric_png_files(folder):
    files = []
    for filename in os.listdir(folder):
        if not filename.endswith(".png"):
            continue
        try:
            frame_index = int(Path(filename).stem)
        except ValueError:
            continue
        files.append((frame_index, folder / filename))
    return sorted(files)


def clamp01(value):
    return float(np.clip(value, 0.0, 1.0))


def ramp(value, low, high):
    if high == low:
        return 0.0
    return clamp01((value - low) / (high - low))


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)


def analyze_frame(path):
    image = load_image(path)
    bgr = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1:] /= 255.0

    center = np.s_[gray.shape[0] // 4 : 3 * gray.shape[0] // 4, gray.shape[1] // 4 : 3 * gray.shape[1] // 4]
    center_gray = gray[center]
    center_hsv = hsv[center]

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 45, 110)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    center_contrast = float(np.std(center_gray))
    sharpness = float(np.var(laplacian))
    edge_density = float(np.mean(edges > 0))
    saturation = float(np.mean(hsv[:, :, 1]))

    too_dark = float(np.mean(gray < 0.08))
    too_bright = float(np.mean(gray > 0.92))
    brightness_extreme = max(ramp(too_dark, 0.15, 0.75), ramp(too_bright, 0.15, 0.75))

    contrast_quality = ramp(contrast, 0.06, 0.20)
    center_contrast_quality = ramp(center_contrast, 0.05, 0.18)
    sharpness_quality = ramp(sharpness, 0.0008, 0.009)
    edge_quality = ramp(edge_density, 0.015, 0.08)
    brightness_quality = 1.0 - brightness_extreme

    visibility_score = clamp01(
        0.28 * contrast_quality
        + 0.22 * center_contrast_quality
        + 0.22 * sharpness_quality
        + 0.18 * edge_quality
        + 0.10 * brightness_quality
    )

    no_vision_score = clamp01(
        0.35 * (1.0 - contrast_quality)
        + 0.25 * (1.0 - sharpness_quality)
        + 0.20 * (1.0 - edge_quality)
        + 0.20 * brightness_extreme
    )

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    brown_mask = (
        (hue >= 0.04)
        & (hue <= 0.16)
        & (sat >= 0.18)
        & (val >= 0.12)
        & (val <= 0.85)
    )
    yellow_mask = (
        (hue > 0.12)
        & (hue <= 0.24)
        & (sat >= 0.14)
        & (val >= 0.18)
        & (val <= 0.95)
    )
    light_deposit_mask = (sat <= 0.24) & (val >= 0.52) & (val <= 0.96)
    dark_structure_mask = (val <= 0.22) & (sat <= 0.65)

    brown_fraction = float(np.mean(brown_mask))
    yellow_fraction = float(np.mean(yellow_mask))
    light_deposit_fraction = float(np.mean(light_deposit_mask))
    dark_structure_fraction = float(np.mean(dark_structure_mask & (edges > 0)))

    brown_deposit_score = clamp01(ramp(brown_fraction + yellow_fraction, 0.10, 0.42))
    light_deposit_score = clamp01(ramp(light_deposit_fraction, 0.20, 0.58))
    dark_structure_score = clamp01(ramp(dark_structure_fraction, 0.003, 0.025))
    dirt_deposit_score = clamp01(
        0.50 * brown_deposit_score
        + 0.35 * light_deposit_score
        + 0.15 * dark_structure_score
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "center_contrast": center_contrast,
        "sharpness": sharpness,
        "edge_density": edge_density,
        "saturation": saturation,
        "visibility_score": visibility_score,
        "no_vision_score": no_vision_score,
        "brown_deposit_score": brown_deposit_score,
        "light_deposit_score": light_deposit_score,
        "dark_structure_score": dark_structure_score,
        "dirt_deposit_score": dirt_deposit_score,
        "brown_fraction": brown_fraction,
        "yellow_fraction": yellow_fraction,
        "light_deposit_fraction": light_deposit_fraction,
        "dark_structure_fraction": dark_structure_fraction,
    }


def moving_average(values, window_size):
    if window_size <= 1 or len(values) == 0:
        return values
    if window_size % 2 == 0:
        window_size += 1
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(values, kernel, mode="same")


def add_smoothed_scores(rows, window_size):
    if not rows:
        return rows
    score_names = [
        "visibility_score",
        "no_vision_score",
        "dirt_deposit_score",
        "brown_deposit_score",
        "light_deposit_score",
        "dark_structure_score",
    ]
    for name in score_names:
        values = np.array([row[name] for row in rows], dtype=np.float32)
        smoothed = moving_average(values, window_size)
        for row, value in zip(rows, smoothed):
            row[f"{name}_smooth"] = float(value)
    return rows


def add_anomaly_score(rows):
    if not rows:
        return rows

    feature_names = [
        "brightness",
        "contrast",
        "sharpness",
        "edge_density",
        "saturation",
        "brown_fraction",
        "light_deposit_fraction",
        "dark_structure_fraction",
    ]
    matrix = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float32)
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    scale = np.maximum(mad * 1.4826, 1e-4)
    robust_distance = np.median(np.abs((matrix - median) / scale), axis=1)
    anomaly = np.array([ramp(value, 1.8, 5.0) for value in robust_distance], dtype=np.float32)

    combined_anomaly = np.maximum.reduce(
        [
            anomaly,
            np.array([row["no_vision_score_smooth"] for row in rows], dtype=np.float32),
            np.array([row["dirt_deposit_score_smooth"] for row in rows], dtype=np.float32) * 0.85,
        ]
    )

    for row, raw_value, combined_value in zip(rows, anomaly, combined_anomaly):
        row["feature_anomaly_score"] = float(raw_value)
        row["anomaly_score"] = float(combined_value)
    return rows


def robust_feature_distance(matrix, reference):
    mad = np.median(np.abs(matrix - reference), axis=0)
    scale = np.maximum(mad * 1.4826, 1e-4)
    distance = np.median(np.abs((matrix - reference) / scale), axis=1)
    return distance


def add_inspection_active_score(rows):
    if not rows:
        return rows

    feature_names = [
        "brightness",
        "contrast",
        "sharpness",
        "edge_density",
        "saturation",
        "brown_fraction",
        "yellow_fraction",
        "light_deposit_fraction",
        "dark_structure_fraction",
    ]
    matrix = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float32)

    start = int(len(rows) * 0.25)
    end = max(start + 1, int(len(rows) * 0.75))
    reference = np.median(matrix[start:end], axis=0)
    distance = robust_feature_distance(matrix, reference)
    pipe_similarity = np.array([1.0 - ramp(value, 1.8, 5.5) for value in distance], dtype=np.float32)

    visibility = np.array([row["visibility_score_smooth"] for row in rows], dtype=np.float32)
    no_vision = np.array([row["no_vision_score_smooth"] for row in rows], dtype=np.float32)
    edge = np.array([row["edge_density"] for row in rows], dtype=np.float32)
    contrast = np.array([row["contrast"] for row in rows], dtype=np.float32)

    texture_quality = np.array(
        [
            0.5 * ramp(edge_value, 0.012, 0.075) + 0.5 * ramp(contrast_value, 0.045, 0.18)
            for edge_value, contrast_value in zip(edge, contrast)
        ],
        dtype=np.float32,
    )

    active = (
        0.50 * pipe_similarity
        + 0.25 * visibility
        + 0.15 * texture_quality
        + 0.10 * (1.0 - no_vision)
    )
    active = np.clip(active, 0.0, 1.0)
    active = moving_average(active, max(5, min(41, len(active) // 8 * 2 + 1)))

    for row, similarity, score in zip(rows, pipe_similarity, active):
        row["pipe_similarity_score"] = float(similarity)
        row["inspection_active_score"] = float(score)
        row["handling_score"] = float(1.0 - score)
    return rows


def longest_active_interval(rows, threshold, min_length):
    candidates = interval_rows(
        rows,
        "inspection_active_score",
        threshold=threshold,
        label="inspection_active",
        min_length=min_length,
    )
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["end_frame"] - row["start_frame"])


def inspection_stage_intervals(rows, min_length):
    if not rows:
        return []

    active = longest_active_interval(rows, threshold=0.58, min_length=min_length)
    if active is None:
        active = {
            "video": rows[0]["video"],
            "start_frame": rows[0]["frame"],
            "end_frame": rows[-1]["frame"],
            "label": "inspection_active",
            "mean_score": float(np.mean([row["inspection_active_score"] for row in rows])),
            "max_score": float(np.max([row["inspection_active_score"] for row in rows])),
        }

    video = rows[0]["video"]
    first_frame = rows[0]["frame"]
    last_frame = rows[-1]["frame"]
    intervals = []

    if active["start_frame"] > first_frame:
        pre_scores = [
            row["handling_score"]
            for row in rows
            if first_frame <= row["frame"] < active["start_frame"]
        ]
        intervals.append(
            {
                "video": video,
                "start_frame": first_frame,
                "end_frame": active["start_frame"] - 1,
                "label": "pre_inspection_handling",
                "mean_score": float(np.mean(pre_scores)) if pre_scores else 0.0,
                "max_score": float(np.max(pre_scores)) if pre_scores else 0.0,
            }
        )

    intervals.append(active)

    if active["end_frame"] < last_frame:
        post_scores = [
            row["handling_score"]
            for row in rows
            if active["end_frame"] < row["frame"] <= last_frame
        ]
        intervals.append(
            {
                "video": video,
                "start_frame": active["end_frame"] + 1,
                "end_frame": last_frame,
                "label": "post_inspection_handling",
                "mean_score": float(np.mean(post_scores)) if post_scores else 0.0,
                "max_score": float(np.max(post_scores)) if post_scores else 0.0,
            }
        )

    return intervals


def interval_rows(rows, score_name, threshold, label, min_length):
    intervals = []
    active = None

    for row in rows:
        is_active = row[score_name] >= threshold
        if is_active and active is None:
            active = {
                "video": row["video"],
                "start_frame": row["frame"],
                "end_frame": row["frame"],
                "label": label,
                "scores": [row[score_name]],
            }
        elif is_active:
            active["end_frame"] = row["frame"]
            active["scores"].append(row[score_name])
        elif active is not None:
            if active["end_frame"] - active["start_frame"] + 1 >= min_length:
                intervals.append(
                    {
                        "video": active["video"],
                        "start_frame": active["start_frame"],
                        "end_frame": active["end_frame"],
                        "label": active["label"],
                        "mean_score": float(np.mean(active["scores"])),
                        "max_score": float(np.max(active["scores"])),
                    }
                )
            active = None

    if active is not None and active["end_frame"] - active["start_frame"] + 1 >= min_length:
        intervals.append(
            {
                "video": active["video"],
                "start_frame": active["start_frame"],
                "end_frame": active["end_frame"],
                "label": active["label"],
                "mean_score": float(np.mean(active["scores"])),
                "max_score": float(np.max(active["scores"])),
            }
        )
    return intervals


def write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_video(video, frame_root, output_root, sample_step, smooth_window, min_interval_frames, frame_index_mode):
    frame_folder = frame_root / str(video)
    if not frame_folder.exists():
        raise FileNotFoundError(f"Missing frame folder: {frame_folder}")

    rows = []
    frame_files = numeric_png_files(frame_folder)
    for ordinal_index, (file_stem_index, path) in enumerate(frame_files):
        if ordinal_index % sample_step != 0:
            continue
        frame_index = file_stem_index if frame_index_mode == "stem" else ordinal_index
        row = {
            "video": video,
            "frame": frame_index,
        }
        row.update(analyze_frame(path))
        rows.append(row)

    add_smoothed_scores(rows, smooth_window)
    add_anomaly_score(rows)
    add_inspection_active_score(rows)

    intervals = []
    intervals.extend(inspection_stage_intervals(rows, min_interval_frames))
    intervals.extend(
        interval_rows(
            rows,
            "no_vision_score_smooth",
            threshold=0.72,
            label="no_vision",
            min_length=min_interval_frames,
        )
    )
    intervals.extend(
        interval_rows(
            rows,
            "dirt_deposit_score_smooth",
            threshold=0.58,
            label="dirt_or_deposit",
            min_length=min_interval_frames,
        )
    )
    intervals.extend(
        interval_rows(
            rows,
            "anomaly_score",
            threshold=0.68,
            label="anomaly_check",
            min_length=min_interval_frames,
        )
    )
    intervals = sorted(intervals, key=lambda row: (row["start_frame"], row["label"]))

    score_path = output_root / "scores" / f"video_{video}_scores.csv"
    interval_path = output_root / "intervals" / f"video_{video}_intervals.csv"
    write_csv(rows, score_path)
    write_csv(intervals, interval_path)

    summary = {
        "video": video,
        "num_sampled_frames": len(rows),
        "sample_step": sample_step,
        "smooth_window": smooth_window,
        "num_intervals": len(intervals),
        "score_path": str(score_path),
        "interval_path": str(interval_path),
    }
    return summary


def available_videos(frame_root):
    return sorted(int(path.name) for path in frame_root.iterdir() if path.is_dir() and path.name.isdigit())


def parse_args():
    parser = argparse.ArgumentParser(description="Heuristic bonus detector for visibility and deposits.")
    parser.add_argument("--video", type=int, default=None, help="Bonus video to analyze. Defaults to all.")
    parser.add_argument(
        "--frame-root",
        type=Path,
        default=DEFAULT_FRAME_ROOT,
        help="Folder containing numbered frame folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Folder where score and interval CSVs are written.",
    )
    parser.add_argument("--sample-step", type=int, default=5, help="Analyze every Nth frame.")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=21,
        help="Moving average window in sampled rows.",
    )
    parser.add_argument(
        "--min-interval-frames",
        type=int,
        default=40,
        help="Discard intervals shorter than this many original frames.",
    )
    parser.add_argument(
        "--frame-index-mode",
        choices=["ordinal", "stem"],
        default="ordinal",
        help="Use ordinal frame positions or PNG filename stems as frame numbers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    videos = [args.video] if args.video is not None else available_videos(args.frame_root)

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for video in videos:
        summary = analyze_video(
            video,
            frame_root=args.frame_root,
            output_root=args.output_root,
            sample_step=args.sample_step,
            smooth_window=args.smooth_window,
            min_interval_frames=args.min_interval_frames,
            frame_index_mode=args.frame_index_mode,
        )
        summaries.append(summary)
        print(
            f"Video {video}: {summary['num_sampled_frames']} sampled frames, "
            f"{summary['num_intervals']} intervals"
        )

    with (args.output_root / "summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)

    print(f"Saved heuristic bonus outputs to {args.output_root}")


if __name__ == "__main__":
    main()
