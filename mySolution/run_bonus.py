import argparse
import csv
from pathlib import Path

import numpy as np

from MovementPathEstimator import MovementPathEstimator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME_ROOT = PROJECT_ROOT / "bonus_frame_images"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "mySolution" / "results" / "saved_bonus_movement_estimations"


def available_videos(frame_root):
    return sorted(int(path.name) for path in frame_root.iterdir() if path.is_dir() and path.name.isdigit())


def save_estimation(video, movement_path, movement_direction, turning_point, channel_length, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"bonus_video_{video}_movement_estimation.npz"
    np.savez_compressed(
        output_path,
        video=int(video),
        channel_length=float(channel_length),
        movement_path=np.asarray(movement_path, dtype=float),
        movement_direction=np.asarray(movement_direction, dtype=float),
        turning_point=float(turning_point),
    )
    return output_path


def channel_length_for_video(video, args):
    if args.use_standard_channel_lengths:
        channel_lengths = np.load(PROJECT_ROOT / "channel_lengths.npy")
        if 1 <= video <= len(channel_lengths):
            return float(channel_lengths[video - 1])
    return float(args.channel_length)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MovementPathEstimator and save bonus-video outputs.")
    parser.add_argument("--video", type=int, default=None, help="Bonus video to run. Defaults to all bonus videos.")
    parser.add_argument("--frame-root", type=Path, default=DEFAULT_FRAME_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--channel-length",
        type=float,
        default=100.0,
        help="Nominal channel length in metres for bonus videos when no true length is available.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Image scale used by MovementPathEstimator. Smaller is faster; standard estimator default is 0.5.",
    )
    parser.add_argument(
        "--use-standard-channel-lengths",
        action="store_true",
        help="Use channel_lengths.npy for bonus videos 1-11. Off by default because bonus lengths are unknown.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    videos = [args.video] if args.video is not None else available_videos(args.frame_root)
    summaries = []

    estimator = MovementPathEstimator(args.video or 1, False)
    estimator.path_to_videos = str(args.frame_root) + "/"
    estimator.scale = float(args.scale)
    estimator.gt_labels = {}
    estimator.suppress_ground_truth_summary = True

    for video in videos:
        channel_length = channel_length_for_video(video, args)
        movement_path, turning_point, movement_direction = estimator.calculate_movement_path_and_turning_point(
            video,
            channel_length,
        )
        output_path = save_estimation(
            video,
            movement_path,
            movement_direction,
            turning_point,
            channel_length,
            args.output_root,
        )
        summaries.append(
            {
                "video": video,
                "num_frames": int(len(movement_path)),
                "channel_length": float(channel_length),
                "turning_point": float(turning_point),
                "max_position": float(np.max(movement_path)),
                "end_position": float(movement_path[-1]),
                "output_path": str(output_path),
            }
        )
        print(f"Saved bonus video {video} estimation to {output_path}")

    summary_path = args.output_root / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video",
                "num_frames",
                "channel_length",
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
