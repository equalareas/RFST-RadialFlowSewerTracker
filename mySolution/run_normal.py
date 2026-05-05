import argparse
import csv
from pathlib import Path

import numpy as np

from MovementPathEstimator import MovementPathEstimator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "mySolution" / "results" / "saved_movement_estimations"


def available_videos():
    frame_root = PROJECT_ROOT / "frame_images"
    return sorted(int(path.name) for path in frame_root.iterdir() if path.is_dir() and path.name.isdigit())


def save_estimation(video, movement_path, movement_direction, turning_point, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"video_{video}_movement_estimation.npz"
    np.savez_compressed(
        output_path,
        video=int(video),
        movement_path=np.asarray(movement_path, dtype=float),
        movement_direction=np.asarray(movement_direction, dtype=float),
        turning_point=float(turning_point),
    )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run MovementPathEstimator and save standard-video outputs.")
    parser.add_argument("--video", type=int, default=None, help="Video to run. Defaults to all standard videos.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    videos = [args.video] if args.video is not None else available_videos()
    summaries = []

    for video in videos:
        estimator = MovementPathEstimator(video, False)
        estimator.execute_estimations()
        result = estimator.calculated_movement_paths[video]
        output_path = save_estimation(
            video,
            result.movement_path,
            result.movement_direction,
            result.turning_point,
            args.output_root,
        )
        summaries.append(
            {
                "video": video,
                "num_frames": int(len(result.movement_path)),
                "turning_point": float(result.turning_point),
                "max_position": float(np.max(result.movement_path)),
                "end_position": float(result.movement_path[-1]),
                "output_path": str(output_path),
            }
        )
        print(f"Saved video {video} estimation to {output_path}")

    summary_path = args.output_root / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video",
                "num_frames",
                "turning_point",
                "max_position",
                "end_position",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
