"""
Classical OpenCV Watershed Part Counter Module.

Implements classic computer vision object counting:
Grayscale conversion -> Blur -> Thresholding (Otsu/Adaptive) -> Distance Transform ->
Local Maxima Markers -> Watershed Segmentation -> Connected Components / Contour Filtering.
"""

import cv2
import numpy as np
import colorsys
from typing import Dict, Any, Tuple, List


class CVCounter:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes OpenCV Watershed Counter.

        Args:
            config: Full dictionary from config.yaml or 'cv_watershed' sub-dictionary.
        """
        self.cv_cfg = config.get("cv_watershed", config)

    def count(self, image: np.ndarray) -> Tuple[int, List[np.ndarray], np.ndarray]:
        """
        Detects and counts touching/overlapping parts using Watershed segmentation.

        Args:
            image (np.ndarray): Input preprocessed image in BGR format.

        Returns:
            Tuple containing:
            - count (int): Total number of valid parts counted.
            - valid_contours (list): List of OpenCV contour arrays for detected parts.
            - overlay_image (np.ndarray): Visual output image with colored regions and numbers.
        """
        # Step 1: Grayscale and Gaussian Blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_k = self.cv_cfg.get("blur_kernel_size", 5)
        if blur_k % 2 == 0:
            blur_k += 1
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        # Step 2: Thresholding (Otsu vs Adaptive)
        method = self.cv_cfg.get("threshold_method", "otsu").lower()
        if method == "adaptive":
            block_size = self.cv_cfg.get("adaptive_block_size", 15)
            if block_size % 2 == 0:
                block_size += 1
            c_val = self.cv_cfg.get("adaptive_c", 2)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_val
            )
        else:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Step 3: Morphological Opening to remove noise
        if self.cv_cfg.get("use_morphology", True):
            kernel_sz = self.cv_cfg.get("morph_kernel_size", 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_sz, kernel_sz))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        else:
            opening = thresh

        # Step 4: Background region identification (sure background)
        sure_bg = cv2.dilate(opening, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=3)

        # Step 5: Distance Transform and Foreground seed identification (sure foreground)
        mask_sz = self.cv_cfg.get("dist_transform_mask_size", 5)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, mask_sz)
        
        factor = float(self.cv_cfg.get("dist_threshold_factor", 0.4))
        _, sure_fg = cv2.threshold(dist_transform, factor * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Step 6: Unknown boundary region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Step 7: Marker labeling for Watershed
        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Step 8: Apply Watershed
        img_for_ws = image.copy()
        markers = cv2.watershed(img_for_ws, markers)

        # Step 9: Extract individual components and apply area/aspect ratio filters
        min_area = self.cv_cfg.get("min_part_area", 150)
        max_area = self.cv_cfg.get("max_part_area", 150000)
        min_ar = self.cv_cfg.get("min_aspect_ratio", 0.2)
        max_ar = self.cv_cfg.get("max_aspect_ratio", 5.0)

        valid_contours = []
        unique_labels = np.unique(markers)

        for label in unique_labels:
            if label <= 1:  # Skip unknown (0) and background (1)
                continue

            component_mask = np.uint8(markers == label) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                if w <= 0 or h <= 0:
                    continue
                aspect_ratio = float(w) / float(h)
                if aspect_ratio < min_ar or aspect_ratio > max_ar:
                    continue

                valid_contours.append(cnt)

        count = len(valid_contours)
        overlay_image = self._create_overlay(image, valid_contours)

        return count, valid_contours, overlay_image

    def _create_overlay(self, image: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
        """Draws colored contour overlays and part indices on image."""
        overlay = image.copy()
        output = image.copy()
        alpha = self.cv_cfg.get("overlay_alpha", 0.45)

        for i, cnt in enumerate(contours, 1):
            # Distinct HSV color
            hue = (i * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            color_bgr = (int(b * 255), int(g * 255), int(r * 255))

            # Draw filled polygon on overlay
            cv2.drawContours(overlay, [cnt], -1, color_bgr, -1)
            cv2.drawContours(output, [cnt], -1, (255, 255, 255), 1)

            # Centroid for label text
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2

            label = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            (w_txt, h_txt), baseline = cv2.getTextSize(label, font, scale, thickness)

            cv2.rectangle(output, (cx - 2, cy - h_txt - 2), (cx + w_txt + 2, cy + baseline + 2), (0, 0, 0), -1)
            cv2.putText(output, label, (cx, cy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output
