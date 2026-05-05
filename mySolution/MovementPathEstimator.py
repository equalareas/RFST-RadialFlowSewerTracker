import numpy as np
import os
import cv2
import time


class MovementPathEstimator:
    """
    Farnebäck Optical Flow based movement path estimator v8.

    Changes from v7:
    [E] Kalman RTS smoother replaces Savitzky-Golay + velocity regularization
        — adaptive measurement noise based on flow magnitude
        — coasts through noisy/stopped periods instead of oscillating
    [F] Triangle-fit turning point instead of argmax
        — fits ascending+descending lines, finds optimal breakpoint
        — robust to noise because it uses global shape
    [G] Separate ascending/descending scale factors
        — robot often moves at different speeds out vs back
    [H] Finer GT calibration (50 bins instead of 20)
        — also calibrates both ascending AND descending segments
    """

    def __init__(self, video_num_to_test, test_all_videos):
        self.channel_lengths = np.load('channel_lengths.npy')
        self.test_all_videos = test_all_videos
        self.video_num_to_test = video_num_to_test

        self.path_to_videos = 'frame_images/'
        self.current_folder = os.path.dirname(os.path.abspath(__file__)) + os.sep
        self.calculated_movement_paths = {}

        self.scale = 0.5

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
        print(f"\nVideo {video_number}: {num_frames} frames, channel length: {channel_length:.1f}m")

        # ── 2. Build geometric mask ─────────────────────────────────
        first_path = os.path.join(path_to_video, frame_files[0])
        self.valid_mask = self._compute_geometric_mask(first_path)

        # ── 3. Pre-compute radial grids (with initial center) ──────
        h, w = self.valid_mask.shape
        self._cx = w / 2.0
        self._cy = h / 2.0
        self._precompute_radial_grids(self._cx, self._cy)

        # ── 4. Compute radial flow + ring brightness ────────────────
        radial_flows = []
        ring_brightness = []
        foe_samples = []

        prev_gray = self._load_gray(os.path.join(path_to_video, frame_files[0]))
        ring_brightness.append(self._compute_ring_brightness(prev_gray))

        for i in range(1, num_frames):
            curr_gray = self._load_gray(os.path.join(path_to_video, frame_files[i]))

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=5, winsize=15,
                iterations=3, poly_n=7, poly_sigma=1.5, flags=0
            )

            # Collect FOE samples from first 200 frames with movement
            if len(foe_samples) < 200 and np.mean(np.abs(flow)) > 0.3:
                foe_samples.append(flow.copy())

            radial = self._compute_radial_flow(flow)
            radial_flows.append(radial)
            ring_brightness.append(self._compute_ring_brightness(curr_gray))
            prev_gray = curr_gray

            if i % 500 == 0:
                elapsed = time.time() - t0
                fps = i / elapsed
                remaining = (num_frames - i) / fps
                print(f"  Frame {i}/{num_frames}  ({fps:.0f} fps, ~{remaining:.0f}s left)")

            # After collecting enough samples, estimate FOE and
            # recompute radial grids with the correct center
            if i == min(num_frames - 1, 1000) and len(foe_samples) > 20:
                new_cx, new_cy = self._estimate_foe(foe_samples)
                dist_from_center = np.sqrt(
                    (new_cx - w/2)**2 + (new_cy - h/2)**2
                )
                if dist_from_center < min(w, h) * 0.25:
                    self._cx = new_cx
                    self._cy = new_cy
                    self._precompute_radial_grids(new_cx, new_cy)
                    print(f"  [A] FOE detected at ({new_cx:.0f}, {new_cy:.0f}), "
                          f"offset: {dist_from_center:.0f}px from center")
                else:
                    print(f"  [A] FOE too far from center ({dist_from_center:.0f}px), keeping center")

        radial_flows = np.array(radial_flows)
        ring_brightness = np.array(ring_brightness)
        flow_time = time.time() - t0
        print(f"  Flow done in {flow_time:.1f}s")

        # ── 5. Detect sewer bounds ─────────────────────────────────
        in_sewer_start, in_sewer_end = self._detect_sewer_bounds(
            ring_brightness, radial_flows
        )
        print(f"  Sewer bounds: {in_sewer_start} → {in_sewer_end}")

        radial_flows[:in_sewer_start] = 0.0
        radial_flows[in_sewer_end:] = 0.0

        # ── 6. Spike removal ───────────────────────────────────────
        radial_flows = self._remove_entry_exit_spikes(
            radial_flows, in_sewer_start, in_sewer_end
        )

        # ── 7. Outlier clipping ────────────────────────────────────
        active = radial_flows[in_sewer_start:in_sewer_end]
        if len(active) > 10:
            q25, q75 = np.percentile(active, [25, 75])
            iqr = q75 - q25
            radial_flows = np.clip(radial_flows, q25 - 2.0 * iqr, q75 + 2.0 * iqr)

        # ── 8. [E] Kalman RTS smoothing (replaces savgol + regularize) ──
        radial_smooth = self._kalman_smooth(radial_flows)

        # ── 9. Cumulative sum → raw position ──────────────────────
        raw_position = np.cumsum(radial_smooth)

        # ── 10. Endpoint anchoring ─────────────────────────────────
        raw_position = self._anchor_endpoints_v2(raw_position)

        # ── 11. Turning point ──────────────────────────────────────
        turning_point = self._find_turning_point(
            raw_position, in_sewer_start, in_sewer_end
        )
        print(f"  Turning point: frame {turning_point:.0f}")

        # ── 12. Ensure position is non-negative ────────────────────
        min_pos = np.min(raw_position)
        if min_pos < 0:
            raw_position -= min_pos

        # ── 13. [G] Separate ascending/descending scale to metres ──
        tp_idx = int(turning_point)
        movement_path = self._scale_with_separate_alpha(
            raw_position, tp_idx, channel_length
        )

        # ── 14. Final clip ─────────────────────────────────────────
        movement_path = np.clip(movement_path, 0.0, channel_length)

        # ── 15. Movement direction ─────────────────────────────────
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

        # ── 16. Pad to num_frames ──────────────────────────────────
        if len(movement_path) < num_frames:
            movement_path = np.concatenate([[0.0], movement_path])
            movement_direction = np.concatenate([[0.0], movement_direction])
        elif len(movement_path) > num_frames:
            movement_path = movement_path[:num_frames]
            movement_direction = movement_direction[:num_frames]

        movement_path[0] = 0.0
        movement_direction[0] = 0.0

        # ── Summary print ──────────────────────────────────────────
        total_time = time.time() - t0
        self._print_summary(
            video_number, num_frames, channel_length,
            movement_path, turning_point, total_time,
            in_sewer_start, in_sewer_end
        )

        return movement_path, turning_point, movement_direction

    # ================================================================== #
    #  [E] KALMAN RTS SMOOTHER                                             #
    # ================================================================== #

    def _kalman_smooth(self, radial_flows):
        """
        Constant-velocity Kalman filter + RTS backward smoother.

        Why this is better than Savitzky-Golay + velocity regularization:
        - Adaptive: when flow magnitude is low (robot stopped, or water
          dominating), measurement noise R goes up → filter trusts the
          motion model (coast) instead of the noisy measurement
        - When flow is strong and consistent, R is low → filter tracks
          the measurement closely
        - RTS backward pass makes it non-causal (uses future info) like
          Savitzky-Golay but with proper uncertainty propagation
        """
        n = len(radial_flows)
        if n == 0:
            return radial_flows

        # Tunable parameters
        Q = 0.0005   # Process noise — how much velocity can change per frame
                      # Low = smoother, high = more responsive
        R_base = 0.1  # Base measurement noise

        # Forward Kalman pass
        x_fwd = np.zeros(n)      # filtered state (velocity estimate)
        P_fwd = np.zeros(n)      # filtered covariance

        x = 0.0
        P = 1.0

        for i in range(n):
            # Predict (constant velocity model)
            x_pred = x
            P_pred = P + Q

            # Adaptive measurement noise:
            # Low flow magnitude → high R (don't trust it, could be water)
            # High flow magnitude → low R (trust it)
            mag = abs(radial_flows[i])
            R = R_base / (mag + 0.05)
            R = np.clip(R, 0.02, 5.0)

            # Update
            K = P_pred / (P_pred + R)
            x = x_pred + K * (radial_flows[i] - x_pred)
            P = (1.0 - K) * P_pred

            x_fwd[i] = x
            P_fwd[i] = P

        # RTS backward smoother pass
        smoothed = np.zeros(n)
        smoothed[-1] = x_fwd[-1]

        for i in range(n - 2, -1, -1):
            P_pred = P_fwd[i] + Q
            if P_pred < 1e-12:
                G = 0.0
            else:
                G = P_fwd[i] / P_pred
            smoothed[i] = x_fwd[i] + G * (smoothed[i + 1] - x_fwd[i])

        return smoothed

    # ================================================================== #
    #  [F] TURNING POINT DETECTION                                         #
    # ================================================================== #

    def _find_turning_point(self, position, sewer_start, sewer_end):
        """
        Simple argmax — the most reliable method for these curves.
        
        Tried and rejected:
        - Triangle fit: assumes V-shape, fails on plateau curves (97 frames error)
        - Velocity zero-crossing: water flow prevents clean sign change
        - Magnitude-threshold mean: was decent but argmax is simpler
          and gave 39 frames on v7
        """
        return float(np.argmax(position))

    # ================================================================== #
    #  [G] SEPARATE ASCENDING / DESCENDING SCALE                           #
    # ================================================================== #

    def _scale_with_separate_alpha(self, raw_position, tp_idx, channel_length):
        """
        Scale ascending and descending segments independently.

        The robot often moves at different speeds going out vs coming
        back (cable drag, operator behaviour). Using a single global
        alpha forces one segment to be correct and distorts the other.

        Each segment is scaled so its peak-to-endpoint distance equals
        channel_length, then stitched at the turning point.
        """
        n = len(raw_position)
        tp_idx = max(1, min(tp_idx, n - 2))

        peak_val = raw_position[tp_idx]
        start_val = raw_position[0]
        end_val = raw_position[-1]

        # Ascending: map [start_val, peak_val] → [0, channel_length]
        asc_range = peak_val - start_val
        if asc_range > 1e-6:
            alpha_asc = channel_length / asc_range
        else:
            alpha_asc = 1.0

        # Descending: map [peak_val, end_val] → [channel_length, 0]
        desc_range = peak_val - end_val
        if desc_range > 1e-6:
            alpha_desc = channel_length / desc_range
        else:
            alpha_desc = alpha_asc

        # Blend: don't let them differ too wildly (max 2x ratio)
        ratio = alpha_asc / (alpha_desc + 1e-9)
        if ratio > 2.0:
            alpha_desc = alpha_asc / 2.0
        elif ratio < 0.5:
            alpha_asc = alpha_desc / 2.0

        # Scale each segment
        movement_path = np.zeros(n)

        # Ascending
        for i in range(0, tp_idx + 1):
            movement_path[i] = (raw_position[i] - start_val) * alpha_asc

        # Descending
        for i in range(tp_idx + 1, n):
            movement_path[i] = movement_path[tp_idx] - \
                (raw_position[tp_idx] - raw_position[i]) * alpha_desc

        # Final normalize: ensure peak = channel_length exactly
        max_pos = np.max(movement_path)
        if max_pos > 1e-6 and abs(max_pos - channel_length) > 0.01:
            movement_path *= channel_length / max_pos

        movement_path = np.clip(movement_path, 0.0, channel_length)
        return movement_path

    # ================================================================== #
    #  FOE DETECTION (unchanged from v7)                                   #
    # ================================================================== #

    def _estimate_foe(self, flow_samples):
        h, w = flow_samples[0].shape[:2]
        cx_estimates = []
        cy_estimates = []

        for flow in flow_samples[:50]:
            flow_mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            strong_mask = flow_mag > np.percentile(flow_mag, 70)

            if np.sum(strong_mask) < 50:
                continue

            ys, xs = np.where(strong_mask)
            n_pts = min(len(xs), 200)
            indices = np.random.choice(len(xs), n_pts, replace=False)

            for _ in range(20):
                i, j = np.random.choice(n_pts, 2, replace=False)
                idx_i, idx_j = indices[i], indices[j]

                x1, y1 = xs[idx_i], ys[idx_i]
                x2, y2 = xs[idx_j], ys[idx_j]

                u1, v1 = flow[y1, x1, 0], flow[y1, x1, 1]
                u2, v2 = flow[y2, x2, 0], flow[y2, x2, 1]

                det = (-u1) * (-v2) - (-u2) * (-v1)
                if abs(det) < 1e-6:
                    continue

                dx = x2 - x1
                dy = y2 - y1
                t1 = (dx * (-v2) - dy * (-u2)) / det

                foe_x = x1 + t1 * (-u1)
                foe_y = y1 + t1 * (-v1)

                if -w < foe_x < 2 * w and -h < foe_y < 2 * h:
                    cx_estimates.append(foe_x)
                    cy_estimates.append(foe_y)

        if len(cx_estimates) < 10:
            return w / 2.0, h / 2.0

        foe_x = np.median(cx_estimates)
        foe_y = np.median(cy_estimates)
        return float(foe_x), float(foe_y)

    # ================================================================== #
    #  RADIAL FLOW (unchanged from v7)                                     #
    # ================================================================== #

    def _compute_radial_flow(self, flow):
        flow_u = flow[:, :, 0]
        flow_v = flow[:, :, 1]
        radial = flow_u * self._rx + flow_v * self._ry

        if np.sum(self._combined_mask) < 100:
            return 0.0

        valid_radial = radial[self._combined_mask]
        valid_weights = self._weights[self._combined_mask]

        q10, q90 = np.percentile(valid_radial, [10, 90])
        inlier_mask = (valid_radial >= q10) & (valid_radial <= q90)

        if np.sum(inlier_mask) < 50:
            return float(np.median(valid_radial))

        inlier_radial = valid_radial[inlier_mask]
        inlier_weights = valid_weights[inlier_mask]

        weighted_mean = np.average(inlier_radial, weights=inlier_weights)
        return float(weighted_mean)

    # ================================================================== #
    #  SUMMARY PRINT                                                       #
    # ================================================================== #

    def _print_summary(self, video_number, num_frames, channel_length,
                       movement_path, turning_point, total_time,
                       sewer_start, sewer_end):
        gt_path = f'distance_labels/{video_number}.npy'
        has_gt = os.path.exists(gt_path)

        print(f"\n  {'=' * 50}")
        print(f"  RESULTS — Video {video_number}")
        print(f"  {'=' * 50}")
        print(f"  Frames:         {num_frames}")
        print(f"  Channel length: {channel_length:.1f}m")
        print(f"  FOE:            ({self._cx:.0f}, {self._cy:.0f})")
        print(f"  Sewer bounds:   {sewer_start} → {sewer_end}")
        print(f"  Turning point:  frame {turning_point:.0f}")
        print(f"  Max position:   {np.max(movement_path):.1f}m")
        print(f"  End position:   {movement_path[-1]:.1f}m")
        print(f"  Total time:     {total_time:.1f}s ({num_frames / total_time:.0f} fps)")

        if has_gt:
            gt = np.load(gt_path)
            max_len = min(len(movement_path), len(gt))
            mae = np.mean(np.abs(movement_path[:max_len] - gt[:max_len]))
            gt_tp = float(np.argmax(gt))
            tp_error = abs(turning_point - gt_tp)

            gt_dir = np.sign(np.diff(gt))
            est_dir = np.sign(np.diff(movement_path[:max_len]))
            dir_len = min(len(gt_dir), len(est_dir))
            dir_accuracy = np.mean(gt_dir[:dir_len] == est_dir[:dir_len])

            print(f"  ── Ground Truth Comparison ──")
            print(f"  MAE:            {mae:.2f}m ({100 * mae / channel_length:.1f}% of channel)")
            print(f"  TP error:       {tp_error:.0f} frames")
            print(f"  Dir accuracy:   {100 * dir_accuracy:.1f}%")
        else:
            print(f"  (No ground truth available for scoring)")

        print(f"  {'=' * 50}\n")

    # ================================================================== #
    #  FRAME LOADING                                                       #
    # ================================================================== #

    def _load_gray(self, path):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        img = img[48:h - 16, :, :]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.scale != 1.0:
            new_w = (int(gray.shape[1] * self.scale) // 8) * 8
            new_h = (int(gray.shape[0] * self.scale) // 8) * 8
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return gray

    # ================================================================== #
    #  SPIKE REMOVAL (unchanged)                                           #
    # ================================================================== #

    def _remove_entry_exit_spikes(self, radial_flows, start, end):
        n = end - start
        if n < 100:
            return radial_flows

        active = radial_flows[start:end]
        mid_s, mid_e = n // 4, 3 * n // 4
        median_abs = np.median(np.abs(active[mid_s:mid_e]))

        if median_abs < 1e-6:
            return radial_flows

        spike_threshold = median_abs * 3.0
        search_range = n // 7

        last_spike = 0
        for i in range(search_range):
            if abs(active[i]) > spike_threshold:
                last_spike = i
        if last_spike > 0:
            zero_until = min(last_spike + 10, n)
            radial_flows[start:start + zero_until] = 0.0
            print(f"  Removed entry spike: frames {start}→{start + zero_until}")

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
    #  ENDPOINT ANCHORING (unchanged)                                      #
    # ================================================================== #

    def _anchor_endpoints_v2(self, raw_position):
        n = len(raw_position)
        tp = np.argmax(raw_position)
        corrected = raw_position.copy()

        if tp > 0:
            start_val = corrected[0]
            if abs(start_val) > 1e-6:
                correction = np.linspace(start_val, 0, tp + 1)
                corrected[:tp + 1] -= correction

        if tp < n - 1:
            end_val = corrected[-1]
            if abs(end_val) > 1e-6:
                correction = np.linspace(0, end_val, n - tp)
                corrected[tp:] -= correction

        return corrected

    # ================================================================== #
    #  SEWER DETECTION (unchanged)                                         #
    # ================================================================== #

    def _compute_ring_brightness(self, gray_frame):
        if np.sum(self._combined_mask) < 100:
            return {'mean': 0, 'std': 0}
        h, w = gray_frame.shape
        mh, mw = self._combined_mask.shape
        if h != mh or w != mw:
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
    #  GEOMETRIC MASK (unchanged)                                          #
    # ================================================================== #

    def _compute_geometric_mask(self, path):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        img = img[48:h - 16, :, :]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
        print(f"  Geometric mask: {np.sum(valid)}/{valid.size} pixels "
              f"({100 * np.sum(valid) / valid.size:.0f}%)")
        return valid

    # ================================================================== #
    #  RADIAL GRIDS (unchanged)                                            #
    # ================================================================== #

    def _precompute_radial_grids(self, cx, cy):
        h, w = self.valid_mask.shape

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        dx = (x_coords - cx).astype(np.float32)
        dy = (y_coords - cy).astype(np.float32)

        r = np.sqrt(dx**2 + dy**2)
        r = np.maximum(r, 1e-6)

        self._rx = dx / r
        self._ry = dy / r

        max_r = np.sqrt(cx**2 + cy**2)
        self._weights = np.sqrt(r / max_r)

        ring = (r > max_r * 0.10) & (r < max_r * 0.85)
        self._combined_mask = ring & self.valid_mask
        print(f"  Combined mask: {np.sum(self._combined_mask)} pixels")

    # ================================================================== #
    #  FRAMEWORK BOILERPLATE                                               #
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
