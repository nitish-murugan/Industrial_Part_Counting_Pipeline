"""
Image Preprocessing Module for Industrial Part Counting.

Handles image loading, aspect-ratio preserving resizing, lighting normalization via CLAHE,
and ROI (Region of Interest) cropping to eliminate background clutter outside the box/tray.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


def apply_clahe(image: np.ndarray, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for lighting normalization.
    Converts color image to LAB space and equalizes the Luminance channel to avoid color distortion.
    """
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl_channel = clahe.apply(l_channel)

    # Merge channels and convert back to BGR
    merged = cv2.merge((cl_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def resize_keep_aspect_ratio(image: np.ndarray, max_dimension: int) -> Tuple[np.ndarray, float]:
    """
    Resizes image so that its maximum dimension does not exceed max_dimension,
    while maintaining the original aspect ratio. Returns resized image and scale factor.
    """
    if max_dimension <= 0:
        return image, 1.0

    h, w = image.shape[:2]
    max_side = max(h, w)

    if max_side <= max_dimension:
        return image, 1.0

    scale = max_dimension / float(max_side)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def crop_roi(image: np.ndarray, roi: Optional[list]) -> np.ndarray:
    """
    Crops Region of Interest from image.
    roi can be [ymin, xmin, ymax, xmax].
    Values can be normalized [0.0 - 1.0] or pixel coordinates.
    """
    if roi is None or len(roi) != 4:
        return image

    h, w = image.shape[:2]
    ymin, xmin, ymax, xmax = roi

    # Convert normalized coordinates if all values are <= 1.0
    if float(ymax) <= 1.0 and float(xmax) <= 1.0:
        ymin = int(ymin * h)
        ymax = int(ymax * h)
        xmin = int(xmin * w)
        xmax = int(xmax * w)
    else:
        ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)

    # Ensure bounds remain valid
    ymin = max(0, min(h - 1, ymin))
    ymax = max(ymin + 1, min(h, ymax))
    xmin = max(0, min(w - 1, xmin))
    xmax = max(xmin + 1, min(w, xmax))

    return image[ymin:ymax, xmin:xmax]


def load_and_preprocess_image(image_path: str, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Loads an image from disk and applies preprocessing based on config parameters.

    Args:
        image_path: Absolute or relative path to the raw input image.
        config: Dictionary loaded from config.yaml under 'preprocessing' key.

    Returns:
        Tuple containing:
        - raw_image (np.ndarray): Original image read from file (BGR).
        - processed_image (np.ndarray): Preprocessed image (BGR, normalized, cropped, resized).
        - scale_factor (float): Scale factor applied during resizing.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise ValueError(f"Failed to decode image from path: {image_path}")

    working_img = raw_image.copy()

    # Step 1: Crop Region of Interest (ROI) if configured
    roi_bounds = config.get("roi_crop", None)
    if roi_bounds is not None:
        working_img = crop_roi(working_img, roi_bounds)

    # Step 2: Resize maintaining aspect ratio
    max_dim = config.get("max_dimension", 1280)
    working_img, scale_factor = resize_keep_aspect_ratio(working_img, max_dim)

    # Step 3: CLAHE Lighting Normalization
    if config.get("use_clahe", True):
        clip_limit = config.get("clahe_clip_limit", 2.5)
        grid_size = tuple(config.get("clahe_tile_grid_size", [8, 8]))
        working_img = apply_clahe(working_img, clip_limit=clip_limit, tile_grid_size=grid_size)

    return raw_image, working_img, scale_factor
