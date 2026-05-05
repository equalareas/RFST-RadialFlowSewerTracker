import numpy as np
import os
import cv2
import time


class MovementPathEstimator:
    """
    Farnebäck Optical Flow based movement path estimator v5.

    Improvements over v4:
    [FIX 1] Downscaling: frames resized to half before flow computation → 2x faster
    [FIX 2] Start spike removal: explicitly detect and zero the insertion spike
    [FIX 3] Stronger drift correction: force start=0 AND end=0
    [FIX 4] Two-pass smoothing: first remove spikes, then smooth properly
    """

    def __init__(self, video_num_to_test, test_all_videos):
        self.channel_lengths = np.load('channel_lengths.npy')
        self.test_all_videos = test_all_videos
        self.video_num_to_test = video_num_to_test

        self.path_to_videos = 'frame_images/'
        self.current_folder = os.path.dirname(os.path.abspath(__file__)) + os.sep
        self.calculated_movement_paths = {}

        # [FIX 1] Downscale factor: 0.5 = half size (4x fewer pixels)
        self.scale = 0.5

        # Load ground truth labels for calibration
        self.gt_labels = {}
        labels_dir = 'distance_labels'
        if os.path.exists(labels_dir):
            for fname in os.listdir(labels_dir):
                if fname.endswith('.npy'):
                    vid_num = int(fname.split('.')[0])
                    self.gt_labels[vid_num] = np.load(
                        os.path.join(labels_dir, fname)
                    )
        print(f"Loaded ground truth for videos: {sorted(self.gt_labels.keys())}")

    # ================================================================== #
    #  MAIN METHOD                                                        #
    # ================================================================== #

    def calculate_movement_path_and_turning_point(self, video_number, channel_length):
        t0 = time.time()
        path_to_video = self.path_to_videos + str(video_number)

        # ── 1. Sorted frame list ────────────────────────────────────
        frame_files = sorted(
            [f for f in os.listdir(path_to_video) if f.endswith('.png')],
            key=lambda x: int(x.split('.')[0])
        )
        num_frames = len(frame_files)
        print(f"Video {video_number}: {num_frames} frames, channel length: {channel_length:.1f}m")

        # ── 2. Build geometric mask ─────────────────────────────────
        first_path = os.path.join(path_to_video, frame_files[0])
        self.valid_mask = self._compute_geometric_mask(first_path)

        # ── 3. Pre-compute radial grids ─────────────────────────────
        self._precompute_radial_grids()

        # ── 4. Compute radial flow + ring brightness ────────────────
        radial_flows = []
        ring_brightness = []

        prev_gray = self._load_gray(os.path.join(path_to_video, frame_files[0]))
        ring_brightness.append(self._compute_ring_brightness(prev_gray))

        for i in range(1, num_frames):
            curr_gray = self._load_gray(os.path.join(path_to_video, frame_files[i]))

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None,
                pyr_scale=0.5,
                levels=5,
                winsize=15,       # Slightly smaller for downscaled images
                iterations=3,
                poly_n=7,
                poly_sigma=1.5,
                flags=0
            )

            radial = self._compute_radial_flow(flow)
            radial_flows.append(radial)
            ring_brightness.append(self._compute_ring_brightness(curr_gray))

            prev_gray = curr_gray

            if i % 500 == 0:
                elapsed = time.time() - t0
                fps = i / elapsed
                remaining = (num_frames - i) / fps
                print(f"  Frame {i}/{num_frames}  ({fps:.0f} fps, ~{remaining:.0f}s left)")

        radial_flows = np.array(radial_flows)
        ring_brightness = np.array(ring_brightness)
        flow_time = time.time() - t0
        print(f"  Flow done in {flow_time:.1f}s. Range: [{radial_flows.min():.4f}, {radial_flows.max():.4f}]")

        # ── 5. Detect sewer bounds ─────────────────────────────────
        in_sewer_start, in_sewer_end = self._detect_sewer_bounds(
            ring_brightness, radial_flows
        )
        print(f"  Sewer bounds: {in_sewer_start} → {in_sewer_end}")

        # Zero out non-sewer
        radial_flows[:in_sewer_start] = 0.0
        radial_flows[in_sewer_end:] = 0.0

        # ── [FIX 2] Spike removal ──────────────────────────────────
        # Even after sewer detection, there can be a spike right when
        # the camera enters. We detect it by finding frames where the
        # absolute flow is much larger than the steady-state flow.
        radial_flows = self._remove_entry_exit_spikes(
            radial_flows, in_sewer_start, in_sewer_end
        )

        # ── 6. Outlier clipping ────────────────────────────────────
        active = radial_flows[in_sewer_start:in_sewer_end]
        if len(active) > 10:
            q25, q75 = np.percentile(active, [25, 75])
            iqr = q75 - q25
            clip_lo = q25 - 2.0 * iqr
            clip_hi = q75 + 2.0 * iqr
            radial_flows = np.clip(radial_flows, clip_lo, clip_hi)

        # ── 7. Savitzky-Golay smoothing ────────────────────────────
        radial_smooth = self._smooth_signal(radial_flows, num_frames)

        # ── 8. Velocity regularization ─────────────────────────────
        radial_smooth = self._regularize_velocity(radial_smooth)

        # ── 9. Cumulative sum → raw position ───────────────────────
        raw_position = np.cumsum(radial_smooth)

        # ── [FIX 3] Stronger endpoint anchoring ───────────────────
        # Force both start ≈ 0 and end ≈ 0.
        raw_position = self._anchor_endpoints_v2(raw_position)

        # ── 10. Turning point ──────────────────────────────────────
        turning_point = float(np.argmax(raw_position))

        # ── 11. Scale to metres ────────────────────────────────────
        min_pos = np.min(raw_position)
        if min_pos < 0:
            raw_position -= min_pos

        max_pos = np.max(raw_position)
        alpha = channel_length / max_pos if max_pos > 1e-6 else 1.0
        movement_path = raw_position * alpha
        movement_path = np.clip(movement_path, 0.0, channel_length)

        # ── 12. Ground truth calibration ───────────────────────────
        movement_path = self._calibrate_with_gt(
            video_number, movement_path, channel_length
        )

        # ── 13. Movement direction ─────────────────────────────────
        path_diff = np.diff(movement_path)
        path_diff = np.concatenate([path_diff, [0.0]])

        active_diff = np.abs(path_diff[in_sewer_start:in_sewer_end])
        if len(active_diff) > 0:
            dir_threshold = np.median(active_diff) * 0.2
        else:
            dir_threshold = 0.001
        dir_threshold = max(dir_threshold, 1e-5)

        movement_direction = np.zeros(len(path_diff))
        movement_direction[path_diff > dir_threshold] = 1.0
        movement_direction[path_diff < -dir_threshold] = -1.0

        # ── 14. Pad to num_frames ──────────────────────────────────
        if len(movement_path) < num_frames:
            movement_path = np.concatenate([[0.0], movement_path])
            movement_direction = np.concatenate([[0.0], movement_direction])
        elif len(movement_path) > num_frames:
            movement_path = movement_path[:num_frames]
            movement_direction = movement_direction[:num_frames]

        movement_path[0] = 0.0
        movement_direction[0] = 0.0

        total_time = time.time() - t0
        print(f"  Turning point: frame {turning_point:.0f}")
        print(f"  Max position:  {np.max(movement_path):.1f}m")
        print(f"  Total time:    {total_time:.1f}s ({num_frames/total_time:.0f} fps)")

        return movement_path, turning_point, movement_direction

    # ================================================================== #
    #  [FIX 1] FRAME LOADING WITH DOWNSCALING                            #
    # ================================================================== #

    def _load_gray(self, path):
        """Load frame as grayscale, crop text, downscale for speed."""
        img = cv2.imread(path)
        h, w = img.shape[:2]
        img = img[48:h-16, :, :]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.scale != 1.0:
            new_w = int(gray.shape[1] * self.scale)
            new_h = int(gray.shape[0] * self.scale)
            # Make dimensions divisible by 8 (some algorithms need this)
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return gray

    # ================================================================== #
    #  [FIX 2] ENTRY/EXIT SPIKE REMOVAL                                  #
    # ================================================================== #

    def _remove_entry_exit_spikes(self, radial_flows, start, end):
        """
        Remove the spike that occurs when the camera first enters or
        exits the sewer. The camera is being pushed/pulled quickly,
        causing extreme flow values for a short period.

        Strategy:
        - Look at the first 15% and last 15% of the active region
        - Compute the median absolute flow of the middle 50%
        - Any frame in those boundary regions with flow > 3× median
          is likely a spike → zero it out
        - Then find where the flow first stabilizes

        This is different from the sewer detection (which detects IF
        we're in the sewer) — this detects the transition WITHIN the
        sewer where the camera is still being positioned.
        """
        n = end - start
        if n < 100:
            return radial_flows

        active = radial_flows[start:end]

        # Reference: median absolute flow in the middle 50%
        mid_s = n // 4
        mid_e = 3 * n // 4
        median_abs = np.median(np.abs(active[mid_s:mid_e]))

        if median_abs < 1e-6:
            return radial_flows

        spike_threshold = median_abs * 3.0

        # Scan the first 15% for spikes
        search_range = n // 7
        last_spike = 0
        for i in range(search_range):
            if abs(active[i]) > spike_threshold:
                last_spike = i

        # Zero out everything up to and slightly past the last spike
        if last_spike > 0:
            zero_until = min(last_spike + 10, n)
            radial_flows[start:start + zero_until] = 0.0
            print(f"  Removed entry spike: frames {start}→{start + zero_until}")

        # Scan the last 15% for spikes
        first_end_spike = n
        for i in range(n - 1, n - search_range, -1):
            if abs(active[i]) > spike_threshold:
                first_end_spike = i

        if first_end_spike < n - 1:
            zero_from = max(first_end_spike - 10, 0)
            radial_flows[start + zero_from:end] = 0.0
            print(f"  Removed exit spike: frames {start + zero_from}→{end}")

        return radial_flows

    # ================================================================== #
    #  [FIX 3] STRONGER ENDPOINT ANCHORING                               #
    # ================================================================== #

    def _anchor_endpoints_v2(self, raw_position):
        """
        Force start=0 and end=0 with a smarter correction.

        Split into two halves at the turning point:
        - Forward half (0 → tp): subtract linear drift scaled so
          that position at frame 0 = 0
        - Backward half (tp → end): subtract linear drift scaled so
          that position at last frame = 0

        This handles each half independently, which is more accurate
        than a single linear correction across the whole video.
        """
        n = len(raw_position)
        tp = np.argmax(raw_position)

        corrected = raw_position.copy()

        # Forward half: should start at 0
        if tp > 0:
            start_val = corrected[0]
            if abs(start_val) > 1e-6:
                # Linear correction from start_val to 0 over forward half
                correction = np.linspace(start_val, 0, tp + 1)
                corrected[:tp+1] -= correction

        # Backward half: should end at 0
        if tp < n - 1:
            end_val = corrected[-1]
            if abs(end_val) > 1e-6:
                # Linear correction from 0 to end_val over backward half
                correction = np.linspace(0, end_val, n - tp)
                corrected[tp:] -= correction

        return corrected

    # ================================================================== #
    #  VISUAL SEWER DETECTION                                             #
    # ================================================================== #

    def _compute_ring_brightness(self, gray_frame):
        if np.sum(self._combined_mask) < 100:
            return {'mean': 0, 'std': 0}
        # Mask might be different size if downscaled
        h, w = gray_frame.shape
        mh, mw = self._combined_mask.shape
        if h != mh or w != mw:
            # Resize mask to match frame
            mask_resized = cv2.resize(
                self._combined_mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST
            ) > 0
            pixels = gray_frame[mask_resized]
        else:
            pixels = gray_frame[self._combined_mask]
        
        if len(pixels) == 0:
            return {'mean': 0, 'std': 0}
        return {'mean': float(np.mean(pixels)), 'std': float(np.std(pixels))}

    def _detect_sewer_bounds(self, ring_brightness, radial_flows):
        n_brightness = len(ring_brightness)
        n_flow = len(radial_flows)

        means = np.array([b['mean'] for b in ring_brightness])
        window = 40
        brightness_stability = np.zeros(n_brightness)
        for i in range(n_brightness):
            lo = max(0, i - window // 2)
            hi = min(n_brightness, i + window // 2)
            brightness_stability[i] = np.std(means[lo:hi])

        within_std = np.array([b['std'] for b in ring_brightness])
        smooth_k = np.ones(21) / 21
        within_std_smooth = np.convolve(within_std, smooth_k, mode='same')

        flow_abs = np.abs(radial_flows)

        mid_s = n_brightness // 5
        mid_e = 4 * n_brightness // 5

        median_brightness_stability = np.median(brightness_stability[mid_s:mid_e])
        median_within_std = np.median(within_std_smooth[mid_s:mid_e])
        median_flow = np.median(flow_abs[mid_s:min(mid_e, n_flow)])

        bright_thresh = max(median_brightness_stability * 3.0, 5.0)
        texture_lo = median_within_std * 0.3
        texture_hi = median_within_std * 2.5
        flow_thresh = max(median_flow * 5.0, 1.0) if median_flow > 1e-6 else float('inf')

        consecutive_needed = 20
        start = 0
        consecutive = 0

        for i in range(min(n_brightness // 3, n_brightness)):
            brightness_ok = brightness_stability[i] < bright_thresh
            texture_ok = texture_lo < within_std_smooth[i] < texture_hi
            flow_ok = flow_abs[i] < flow_thresh if i < n_flow else True

            if brightness_ok and texture_ok and flow_ok:
                consecutive += 1
                if consecutive >= consecutive_needed:
                    start = i - consecutive_needed + 1
                    break
            else:
                consecutive = 0

        end = n_flow
        consecutive = 0
        for i in range(n_brightness - 1, max(2 * n_brightness // 3, 0), -1):
            brightness_ok = brightness_stability[i] < bright_thresh
            texture_ok = texture_lo < within_std_smooth[i] < texture_hi
            flow_ok = flow_abs[i - 1] < flow_thresh if 0 <= i - 1 < n_flow else True

            if brightness_ok and texture_ok and flow_ok:
                consecutive += 1
                if consecutive >= consecutive_needed:
                    end = min(i + consecutive_needed - 1, n_flow)
                    break
            else:
                consecutive = 0

        if end - start < n_flow * 0.5:
            start, end = 0, n_flow

        return start, end

    # ================================================================== #
    #  SMOOTHING                                                          #
    # ================================================================== #

    def _smooth_signal(self, signal, num_frames):
        try:
            from scipy.signal import savgol_filter
            window = max(31, min(101, num_frames // 40))
            window = window if window % 2 == 1 else window + 1
            return savgol_filter(signal, window_length=window, polyorder=3)
        except ImportError:
            kernel_size = max(15, min(51, num_frames // 80))
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            kernel = np.ones(kernel_size) / kernel_size
            return np.convolve(signal, kernel, mode='same')

    # ================================================================== #
    #  VELOCITY REGULARIZATION                                            #
    # ================================================================== #

    def _regularize_velocity(self, flow_signal):
        acceleration = np.diff(flow_signal)
        acc_kernel = np.ones(41) / 41
        acceleration_smooth = np.convolve(acceleration, acc_kernel, mode='same')

        regularized = np.zeros_like(flow_signal)
        regularized[0] = flow_signal[0]
        for i in range(1, len(flow_signal)):
            regularized[i] = regularized[i-1] + acceleration_smooth[i-1]

        return 0.7 * regularized + 0.3 * flow_signal

    # ================================================================== #
    #  GT CALIBRATION                                                     #
    # ================================================================== #

    def _calibrate_with_gt(self, video_number, movement_path, channel_length):
        if len(self.gt_labels) == 0:
            return movement_path

        n = len(movement_path)
        num_bins = 20
        correction_profiles = []

        for vid_num, gt in self.gt_labels.items():
            if vid_num == video_number:
                continue
            gt_len = len(gt)
            gt_norm = gt / max(np.max(gt), 1e-6)
            bins = np.linspace(0, gt_len - 1, num_bins + 1).astype(int)
            profile = []
            for b in range(num_bins):
                s, e = bins[b], bins[b + 1]
                if e > s and e < gt_len:
                    profile.append(gt_norm[e] - gt_norm[s])
                else:
                    profile.append(0)
            profile = np.array(profile)
            total = np.sum(np.abs(profile))
            if total > 1e-6:
                correction_profiles.append(profile / total)

        if not correction_profiles:
            return movement_path

        avg_profile = np.mean(correction_profiles, axis=0)
        tp = int(np.argmax(movement_path))
        max_val = np.max(movement_path)

        if tp > 10 and max_val > 1e-6:
            our_bins = np.linspace(0, tp, num_bins + 1).astype(int)
            path_norm = movement_path[:tp+1] / max_val
            our_profile = []
            for b in range(num_bins):
                s, e = our_bins[b], our_bins[b+1]
                if e > s and e <= tp:
                    our_profile.append(path_norm[e] - path_norm[s])
                else:
                    our_profile.append(0)
            our_profile = np.array(our_profile)
            our_total = np.sum(np.abs(our_profile))

            if our_total > 1e-6:
                our_norm = our_profile / our_total
                ratio = np.ones(num_bins)
                for b in range(num_bins):
                    if our_norm[b] > 1e-6:
                        ratio[b] = 0.7 + 0.3 * (avg_profile[b] / our_norm[b])
                ratio = np.clip(ratio, 0.5, 2.0)

                corrected = np.copy(movement_path[:tp+1])
                for b in range(num_bins):
                    s, e = our_bins[b], our_bins[b+1]
                    if e > s and e <= tp:
                        seg = corrected[s:e+1]
                        inc = np.diff(seg) * ratio[b]
                        corrected[s+1:e+1] = seg[0] + np.cumsum(inc)

                if np.max(corrected) > 1e-6:
                    corrected *= max_val / np.max(corrected)
                movement_path[:tp+1] = corrected

        return movement_path

    # ================================================================== #
    #  GEOMETRIC MASK                                                     #
    # ================================================================== #

    def _compute_geometric_mask(self, path):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        img = img[48:h-16, :, :]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply same downscaling as frames
        if self.scale != 1.0:
            new_w = (int(gray.shape[1] * self.scale) // 8) * 8
            new_h = (int(gray.shape[0] * self.scale) // 8) * 8
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        morph_k = np.ones((9, 9), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_k)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones(gray.shape, dtype=bool)

        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

        erode_k = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, erode_k, iterations=1)

        valid = mask > 0
        print(f"  Geometric mask: {np.sum(valid)}/{valid.size} pixels ({100*np.sum(valid)/valid.size:.0f}%)")
        return valid

    # ================================================================== #
    #  RADIAL GRIDS                                                       #
    # ================================================================== #

    def _precompute_radial_grids(self):
        h, w = self.valid_mask.shape
        cx, cy = w / 2.0, h / 2.0

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        dx = (x_coords - cx).astype(np.float32)
        dy = (y_coords - cy).astype(np.float32)

        r = np.sqrt(dx**2 + dy**2)
        r = np.maximum(r, 1e-6)

        self._rx = dx / r
        self._ry = dy / r

        max_r = np.sqrt(cx**2 + cy**2)
        ring = (r > max_r * 0.10) & (r < max_r * 0.85)
        self._combined_mask = ring & self.valid_mask
        print(f"  Combined mask: {np.sum(self._combined_mask)} pixels")

    # ================================================================== #
    #  RADIAL FLOW                                                        #
    # ================================================================== #

    def _compute_radial_flow(self, flow):
        flow_u = flow[:, :, 0]
        flow_v = flow[:, :, 1]
        radial = flow_u * self._rx + flow_v * self._ry

        if np.sum(self._combined_mask) < 100:
            return 0.0
        return np.mean(radial[self._combined_mask])

    # ================================================================== #
    #  FRAMEWORK BOILERPLATE                                              #
    # ================================================================== #

    def execute_estimations(self):
        if self.test_all_videos:
            if not os.path.exists(self.path_to_videos):
                raise FileNotFoundError(
                    f"The folder '{self.path_to_videos}' does not exist."
                )
            for entry in os.listdir(self.path_to_videos):
                if entry.isdigit():
                    self._run_single(int(entry))
        else:
            self._run_single(self.video_num_to_test)

    def _run_single(self, video_number):
        try:
            channel_length = self.channel_lengths[video_number - 1]
        except Exception:
            print("Cannot load channel length, using 100 m")
            channel_length = 100
        movement_path, turning_point, movement_direction = \
            self.calculate_movement_path_and_turning_point(
                int(video_number), channel_length
            )
        self.calculated_movement_paths[int(video_number)] = MovementPath(
            int(video_number), movement_path, movement_direction, turning_point
        )


from MovementPath import MovementPath