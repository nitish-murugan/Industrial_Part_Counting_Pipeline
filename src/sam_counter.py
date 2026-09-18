"""
SAM-based Part Counter Module (Segment Anything Model).

Uses SAM's automatic mask generator (via segment-anything or ultralytics) to extract object masks,
filters candidate masks by area, aspect ratio, and quality heuristics loaded from config.yaml,
and produces a visual overlay with numbered colored masks.
"""

import os
import torch
import numpy as np
import cv2
import colorsys
from typing import Dict, Any, Tuple, List, Optional


class SAMCounter:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the SAM Counter with configuration settings.

        Args:
            config: Full dictionary from config.yaml or 'sam' sub-dictionary.
        """
        self.sam_cfg = config.get("sam", config)
        self.paths_cfg = config.get("paths", {})
        self.backend = self.sam_cfg.get("backend", "segment-anything")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mask_generator = None
        self._init_model()

    def _init_model(self):
        """Lazy loads SAM model according to backend setting."""
        checkpoint = self.paths_cfg.get("sam_checkpoint", self.sam_cfg.get("sam_checkpoint", ""))
        model_type = self.paths_cfg.get("model_type", self.sam_cfg.get("model_type", "vit_h"))

        if self.backend == "ultralytics":
            try:
                from ultralytics import SAM
                model_name = "sam2_b.pt" if not os.path.exists(checkpoint) else checkpoint
                self.model = SAM(model_name)
                print(f"[SAMCounter] Loaded Ultralytics SAM ({model_name}) on {self.device}")
            except ImportError:
                print("[SAMCounter] Ultralytics package not available. Falling back to segment-anything.")
                self.backend = "segment-anything"

        if self.backend == "segment-anything":
            try:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                if not os.path.exists(checkpoint):
                    print(f"[SAMCounter WARNING] SAM checkpoint file not found at '{checkpoint}'. "
                          f"Please download the checkpoint file when running in Colab.")
                    self.mask_generator = None
                    return

                sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
                sam_model.to(device=self.device)

                self.mask_generator = SamAutomaticMaskGenerator(
                    model=sam_model,
                    points_per_side=int(self.sam_cfg.get("points_per_side", 32)),
                    pred_iou_thresh=float(self.sam_cfg.get("pred_iou_thresh", 0.86)),
                    stability_score_thresh=float(self.sam_cfg.get("stability_score_thresh", 0.90)),
                    min_mask_region_area=int(self.sam_cfg.get("min_mask_area", 200)),
                    box_nms_thresh=float(self.sam_cfg.get("box_nms_thresh", 0.7)),
                    crop_n_layers=int(self.sam_cfg.get("crop_n_layers", 0)),
                )
                print(f"[SAMCounter] Loaded segment-anything ({model_type}) on {self.device}")
            except Exception as e:
                print(f"[SAMCounter ERROR] Failed to initialize SAM: {e}")
                self.mask_generator = None

    def filter_mask(self, mask_dict: Dict[str, Any], img_shape: Tuple[int, int]) -> bool:
        """
        Applies configured area and aspect ratio heuristics to filter out noise,
        background/box regions, and non-part mask candidates.
        """
        area = mask_dict.get("area", np.sum(mask_dict["segmentation"]))
        min_area = self.sam_cfg.get("min_mask_area", 200)
        max_area = self.sam_cfg.get("max_mask_area", 150000)

        if area < min_area or area > max_area:
            return False

        # Bounding box filtering: bbox format is [x, y, w, h]
        bbox = mask_dict.get("bbox", None)
        if bbox is not None:
            _, _, w, h = bbox
            if w <= 0 or h <= 0:
                return False
            aspect_ratio = float(w) / float(h)
            min_ar = self.sam_cfg.get("min_aspect_ratio", 0.2)
            max_ar = self.sam_cfg.get("max_aspect_ratio", 5.0)

            if aspect_ratio < min_ar or aspect_ratio > max_ar:
                return False

        return True

    def count(self, image: np.ndarray) -> Tuple[int, List[Dict[str, Any]], np.ndarray]:
        """
        Detects and counts individual parts in the image using SAM.

        Args:
            image (np.ndarray): Input preprocessed image in BGR format.

        Returns:
            Tuple containing:
            - count (int): Total number of valid parts counted.
            - filtered_masks (list): List of accepted mask dictionaries.
            - overlay_image (np.ndarray): Visual output image with colored masks & indices.
        """
        if self.backend == "ultralytics":
            return self._count_ultralytics(image)

        if self.mask_generator is None:
            print("[SAMCounter WARNING] SAM model generator is uninitialized. Returning count=0.")
            return 0, [], image.copy()

        # Convert image to RGB for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_masks = self.mask_generator.generate(image_rgb)

        img_h, img_w = image.shape[:2]
        filtered_masks = [m for m in raw_masks if self.filter_mask(m, (img_h, img_w))]

        # Sort masks by area descending
        filtered_masks.sort(key=lambda x: x["area"], reverse=True)

        count = len(filtered_masks)
        overlay_image = self._create_overlay(image, filtered_masks)

        return count, filtered_masks, overlay_image

    def _count_ultralytics(self, image: np.ndarray) -> Tuple[int, List[Dict[str, Any]], np.ndarray]:
        """Alternative ultralytics SAM mask generator count path."""
        results = self.model(image, verbose=False)
        filtered_masks = []
        img_h, img_w = image.shape[:2]

        if results and len(results) > 0 and results[0].masks is not None:
            masks_data = results[0].masks.data.cpu().numpy()
            for mask_arr in masks_data:
                seg = mask_arr.astype(bool)
                area = np.sum(seg)
                y_idx, x_idx = np.where(seg)
                if len(x_idx) == 0 or len(y_idx) == 0:
                    continue
                bbox = [np.min(x_idx), np.min(y_idx), np.max(x_idx) - np.min(x_idx), np.max(y_idx) - np.min(y_idx)]
                m_dict = {"segmentation": seg, "area": area, "bbox": bbox}
                if self.filter_mask(m_dict, (img_h, img_w)):
                    filtered_masks.append(m_dict)

        count = len(filtered_masks)
        overlay_image = self._create_overlay(image, filtered_masks)
        return count, filtered_masks, overlay_image

    def _create_overlay(self, image: np.ndarray, masks: List[Dict[str, Any]]) -> np.ndarray:
        """Draws distinct colored masks and part numbers over the input image."""
        overlay = image.copy()
        output = image.copy()
        alpha = self.sam_cfg.get("overlay_alpha", 0.45)

        for i, mask_info in enumerate(masks, 1):
            seg = mask_info["segmentation"]
            
            # Generate distinct HSV color per index
            hue = (i * 0.618033988749895) % 1.0  # Golden ratio color distribution
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            color_bgr = (int(b * 255), int(g * 255), int(r * 255))

            # Fill mask color
            overlay[seg] = color_bgr

            # Find mask centroid for label positioning
            y_indices, x_indices = np.where(seg)
            if len(x_indices) > 0 and len(y_indices) > 0:
                cx, cy = int(np.mean(x_indices)), int(np.mean(y_indices))

                # Draw contour outline
                mask_uint8 = seg.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contours, -1, (255, 255, 255), 1)

                # Draw index number with background pill for readability
                label = str(i)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 1
                (w, h), baseline = cv2.getTextSize(label, font, scale, thickness)

                cv2.rectangle(output, (cx - 2, cy - h - 2), (cx + w + 2, cy + baseline + 2), (0, 0, 0), -1)
                cv2.putText(output, label, (cx, cy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Blend original image and color overlay
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output
