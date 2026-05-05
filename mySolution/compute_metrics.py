import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ESTIMATION_ROOT = PROJECT_ROOT / "mySolution" / "results" / "saved_movement_estimations"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "mySolution" / "results" / "saved_movement_estimations" / "evaluation.csv"


def load_estimation(path):
    data = np.load(path)
    return {
        "video": int(data["video"]),
        "movement_path": np.asarray(data["movement_path"], dtype=float),
        "movement_direction": np.asarray(data["movement_direction"], dtype=float),
        "turning_point": float(data["turning_point"]),
    }


def movement_direction(path):
    diff = np.diff(path)
    threshold = max(float(np.nanmax(np.abs(diff))) * 0.02, 1e-6)
    direction = np.zeros(path.shape[0], dtype=float)
    direction[:-1][diff > threshold] = 1.0
    direction[:-1][diff < -threshold] = -1.0
    return direction


def evaluate_video(estimation, channel_length):
    video = estimation["video"]
    label_path = PROJECT_ROOT / "distance_labels" / f"{video}.npy"
    if not label_path.exists():
        return {
            "video": video,
            "has_ground_truth": False,
            "num_frames_compared": "",
            "path_mae_m": "",
            "path_mae_normalized": "",
            "turning_point_estimated": estimation["turning_point"],
            "turning_point_measured": "",
            "turning_point_error_frames": "",
            "direction_accuracy": "",
        }

    measured = np.load(label_path).astype(float)
    estimated = estimation["movement_path"]
    n = min(len(measured), len(estimated))
    measured = measured[:n]
    estimated = estimated[:n]

    path_mae = float(np.mean(np.abs(estimated - measured)))
    measured_tp = float(np.argmax(measured))
    tp_error = float(abs(estimation["turning_point"] - measured_tp))

    estimated_dir = movement_direction(estimated)
    measured_dir = movement_direction(measured)
    direction_accuracy = float(np.mean(estimated_dir[:n] == measured_dir[:n]))

    return {
        "video": video,
        "has_ground_truth": True,
        "num_frames_compared": n,
        "path_mae_m": path_mae,
        "path_mae_normalized": path_mae / float(channel_length),
        "turning_point_estimated": estimation["turning_point"],
        "turning_point_measured": measured_tp,
        "turning_point_error_frames": tp_error,
        "direction_accuracy": direction_accuracy,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved movement estimations against available labels.")
    parser.add_argument("--estimation-root", type=Path, default=DEFAULT_ESTIMATION_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    channel_lengths = np.load(PROJECT_ROOT / "channel_lengths.npy")
    rows = []

    for path in sorted(args.estimation_root.glob("video_*_movement_estimation.npz")):
        estimation = load_estimation(path)
        channel_length = channel_lengths[estimation["video"] - 1]
        rows.append(evaluate_video(estimation, channel_length))

    labeled = [row for row in rows if row["has_ground_truth"]]
    if labeled:
        rows.append(
            {
                "video": "average_labeled",
                "has_ground_truth": True,
                "num_frames_compared": "",
                "path_mae_m": float(np.mean([row["path_mae_m"] for row in labeled])),
                "path_mae_normalized": float(np.mean([row["path_mae_normalized"] for row in labeled])),
                "turning_point_estimated": "",
                "turning_point_measured": "",
                "turning_point_error_frames": float(np.mean([row["turning_point_error_frames"] for row in labeled])),
                "direction_accuracy": float(np.mean([row["direction_accuracy"] for row in labeled])),
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video",
                "has_ground_truth",
                "num_frames_compared",
                "path_mae_m",
                "path_mae_normalized",
                "turning_point_estimated",
                "turning_point_measured",
                "turning_point_error_frames",
                "direction_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved evaluation to {args.output_path}")
    for row in rows:
        if not row["has_ground_truth"]:
            print(f"Video {row['video']}: no ground truth")
            continue
        print(
            f"Video {row['video']}: "
            f"MAE={row['path_mae_m']:.2f} m, "
            f"normMAE={row['path_mae_normalized']:.3f}, "
            f"TP error={row['turning_point_error_frames']:.1f} frames, "
            f"dir acc={100 * row['direction_accuracy']:.1f}%"
        )


if __name__ == "__main__":
    main()
