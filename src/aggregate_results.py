"""
Results Aggregation Module.

Processes single images or entire folders of raw images in batch mode,
runs requested part counting algorithms (SAM, OpenCV Watershed),
saves visual overlay output images, and writes a summary CSV file.
"""

import os
import time
import glob
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from .preprocess import load_and_preprocess_image
from .sam_counter import SAMCounter
from .cv_counter import CVCounter


class ResultsAggregator:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the aggregator with configuration parameters.

        Args:
            config: Full dictionary loaded from config.yaml.
        """
        self.config = config
        self.paths_cfg = config.get("paths", {})
        self.output_dir = self.paths_cfg.get("output_dir", "output_results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.sam_counter = None
        self.cv_counter = None

    def _get_image_paths(self, input_path: str) -> List[str]:
        """Collects list of image file paths from a file or folder path."""
        if os.path.isfile(input_path):
            return [input_path]

        if os.path.isdir(input_path):
            extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.NIGHT.jpg", "*.MP.jpg"]
            image_paths = []
            for ext in extensions:
                image_paths.extend(glob.glob(os.path.join(input_path, ext)))
                image_paths.extend(glob.glob(os.path.join(input_path, ext.upper())))
            return sorted(list(set(image_paths)))

        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    def run(self, input_path: str, mode: str = "all") -> pd.DataFrame:
        """
        Executes part counting on target image(s).

        Args:
            input_path: Path to single image or folder of images.
            mode: Operating mode - 'sam', 'cv', or 'all'.

        Returns:
            pd.DataFrame: Summary dataframe with counts per image.
        """
        image_paths = self._get_image_paths(input_path)
        print(f"[ResultsAggregator] Found {len(image_paths)} images to process.")

        # Initialize requested counters lazily
        if mode in ["sam", "all"]:
            self.sam_counter = SAMCounter(self.config)
        if mode in ["cv", "all"]:
            self.cv_counter = CVCounter(self.config)

        results = []

        for idx, img_path in enumerate(image_paths, 1):
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            print(f"[{idx}/{len(image_paths)}] Processing {filename}...")

            start_time = time.time()

            try:
                raw_img, processed_img, scale_factor = load_and_preprocess_image(
                    img_path, self.config.get("preprocessing", {})
                )
            except Exception as e:
                print(f"[ERROR] Failed to preprocess {filename}: {e}")
                continue

            sam_count = None
            cv_count = None

            sam_overlay = None
            cv_overlay = None

            # Subfolder per image to store overlays
            img_out_dir = os.path.join(self.output_dir, base_name)
            os.makedirs(img_out_dir, exist_ok=True)

            # Save preprocessed input image
            cv2.imwrite(os.path.join(img_out_dir, "processed_input.jpg"), processed_img)

            # Run SAM Counter
            if mode in ["sam", "all"] and self.sam_counter is not None:
                sam_count, _, sam_overlay = self.sam_counter.count(processed_img)
                cv2.imwrite(os.path.join(img_out_dir, "sam_overlay.jpg"), sam_overlay)

            # Run OpenCV Watershed Counter
            if mode in ["cv", "all"] and self.cv_counter is not None:
                cv_count, _, cv_overlay = self.cv_counter.count(processed_img)
                cv2.imwrite(os.path.join(img_out_dir, "cv_overlay.jpg"), cv_overlay)

            # Create side-by-side visual comparison in 'all' mode
            if mode == "all" and sam_overlay is not None and cv_overlay is not None:
                sbs_comparison = self._create_side_by_side(
                    processed_img, sam_overlay, cv_overlay, sam_count, cv_count
                )
                cv2.imwrite(os.path.join(img_out_dir, "comparison_side_by_side.jpg"), sbs_comparison)

            proc_time = round(time.time() - start_time, 2)

            results.append({
                "filename": filename,
                "sam_count": sam_count,
                "cv_count": cv_count,
                "processing_time_sec": proc_time,
                "scale_factor": round(scale_factor, 3),
                "notes": "Success"
            })

            print(f" -> Results for {filename}: SAM Count={sam_count}, CV Count={cv_count} ({proc_time}s)")

        # Create and export Summary CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, "summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[ResultsAggregator] Summary exported to '{csv_path}'")

        return df

    def _create_side_by_side(
        self, raw: np.ndarray, sam_img: np.ndarray, cv_img: np.ndarray, sam_cnt: int, cv_cnt: int
    ) -> np.ndarray:
        """Combines original, SAM overlay, and CV Watershed overlay into a labeled side-by-side image."""
        h, w = raw.shape[:2]
        banner_h = 40
        canvas = np.zeros((h + banner_h, w * 2, 3), dtype=np.uint8)

        # Place SAM overlay on left, CV overlay on right
        canvas[banner_h:, :w] = sam_img
        canvas[banner_h:, w:] = cv_img

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"SAM Model (Count: {sam_cnt})", (15, 26), font, 0.7, (0, 255, 255), 2)
        cv2.putText(canvas, f"OpenCV Watershed (Count: {cv_cnt})", (w + 15, 26), font, 0.7, (255, 200, 0), 2)

        return canvas
