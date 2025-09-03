import numpy as np
import cv2


class Blender:
    def linearBlending(self, imgs):
        '''
        linear Blending(also known as Feathering)
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]

        # Non-black masks (use OpenCV for speed)
        left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        left_mask = left_gray > 0
        right_mask = right_gray > 0
        overlap = left_mask & right_mask

        # Removed per-frame plotting for performance in real-time video

        # If no overlap, simply composite
        if not np.any(overlap):
            out = img_right.copy()
            out[left_mask] = img_left[left_mask]
            return out

        # Restrict computation to overlap ROI
        coords = cv2.findNonZero(overlap.astype(np.uint8))
        if coords is None:
            out = img_right.copy()
            out[left_mask] = img_left[left_mask]
            return out
        x, y, rw, rh = cv2.boundingRect(coords)

        overlap_roi = overlap[y:y+rh, x:x+rw]
        row_has_overlap = np.any(overlap_roi, axis=1)
        first = np.zeros(rh, dtype=int)
        last = np.zeros(rh, dtype=int)
        if np.any(row_has_overlap):
            first[row_has_overlap] = np.argmax(overlap_roi[row_has_overlap], axis=1)
            last[row_has_overlap] = (rw - 1) - np.argmax(overlap_roi[row_has_overlap][:, ::-1], axis=1)

        length = np.maximum(1, (last - first))
        j = np.arange(rw)[None, :]
        first_col = first[:, None]
        length_col = length[:, None]

        alpha_roi = np.zeros((rh, rw), dtype=np.float32)
        inside = (j >= first_col) & (j <= (last[:, None])) & row_has_overlap[:, None]
        ramp = 1.0 - (j - first_col) / length_col
        alpha_roi[inside] = ramp[inside]

        out = img_right.astype(np.float32).copy()
        only_left = left_mask & ~right_mask
        out[only_left] = img_left[only_left]

        # Blend only within overlap ROI
        if np.any(row_has_overlap):
            ys, xs = np.where(overlap_roi)
            y_idx = y + ys
            x_idx = x + xs
            a_vals = alpha_roi[ys, xs][:, None]
            out[y_idx, x_idx] = a_vals * img_left[y_idx, x_idx] + (1.0 - a_vals) * img_right[y_idx, x_idx]

        return np.clip(out, 0, 255).astype(np.uint8)

    def linearBlendingWithConstantWidth(self, imgs):
        '''
        linear Blending with Constat Width, avoiding ghost region
        # you need to determine the size of constant with
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        constant_width = 3

        # Non-black masks (use OpenCV for speed)
        left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        left_mask = left_gray > 0
        right_mask = right_gray > 0
        overlap = left_mask & right_mask

        if not np.any(overlap):
            out = img_right.copy()
            out[left_mask] = img_left[left_mask]
            return out

        coords = cv2.findNonZero(overlap.astype(np.uint8))
        if coords is None:
            out = img_right.copy()
            out[left_mask] = img_left[left_mask]
            return out
        x, y, rw, rh = cv2.boundingRect(coords)

        overlap_roi = overlap[y:y+rh, x:x+rw]
        row_has_overlap = np.any(overlap_roi, axis=1)
        first = np.zeros(rh, dtype=int)
        last = np.zeros(rh, dtype=int)
        if np.any(row_has_overlap):
            first[row_has_overlap] = np.argmax(overlap_roi[row_has_overlap], axis=1)
            last[row_has_overlap] = (rw - 1) - np.argmax(overlap_roi[row_has_overlap][:, ::-1], axis=1)

        mid = (first + last) // 2
        j = np.arange(rw)[None, :]
        mid_col = mid[:, None]
        first_col = first[:, None]
        last_col = last[:, None]

        alpha_roi = np.zeros((rh, rw), dtype=np.float32)
        # Regions fully left/right of the blending band
        alpha_roi = np.where(j <= (mid_col - constant_width), 1.0, alpha_roi)
        alpha_roi = np.where(j >= (mid_col + constant_width), 0.0, alpha_roi)

        # Linear ramp inside the band
        band = (j > (mid_col - constant_width)) & (j < (mid_col + constant_width)) & row_has_overlap[:, None]
        alpha_band = 1.0 - (j - (mid_col - constant_width)) / (2.0 * constant_width)
        alpha_roi = np.where(band, alpha_band, alpha_roi)

        # Zero out columns outside the overlap bounds
        valid = (j >= first_col) & (j <= last_col) & row_has_overlap[:, None]
        alpha_roi = np.where(valid, alpha_roi, 0.0)

        out = img_right.astype(np.float32).copy()
        only_left = left_mask & ~right_mask
        out[only_left] = img_left[only_left]

        if np.any(row_has_overlap):
            ys, xs = np.where(overlap_roi)
            y_idx = y + ys
            x_idx = x + xs
            a_vals = alpha_roi[ys, xs][:, None]
            out[y_idx, x_idx] = a_vals * img_left[y_idx, x_idx] + (1.0 - a_vals) * img_right[y_idx, x_idx]

        return np.clip(out, 0, 255).astype(np.uint8)