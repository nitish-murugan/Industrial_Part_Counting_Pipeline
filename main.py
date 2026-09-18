import sys
import os
import argparse
import yaml
from typing import Dict, Any

# Ensure project root directory is in Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.aggregate_results import ResultsAggregator
    from src.evaluate import evaluate_predictions
except ImportError:
    from aggregate_results import ResultsAggregator
    from evaluate import evaluate_predictions


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Shot Industrial Part Counting Pipeline (SAM & OpenCV Watershed)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to single input image or dataset directory containing raw images.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file containing tunable thresholds.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sam", "cv", "all"],
        default="all",
        help="Part counting method mode: 'sam', 'cv' (Watershed), or 'all' (compare both).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory where output overlays and summary.csv will be saved.",
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        default=None,
        help="Optional path to ground_truth_counts.csv for evaluating MAE/RMSE metrics.",
    )

    args = parser.parse_args()

    # Load YAML Configuration
    config = load_config(args.config)

    # CLI Overrides
    input_path = args.input if args.input else config.get("paths", {}).get("input_dir", "dataset")
    if args.output:
        if "paths" not in config:
            config["paths"] = {}
        config["paths"]["output_dir"] = args.output

    print("=================================================================")
    print("      UNSUPERVISED INDUSTRIAL PART COUNTING PIPELINE             ")
    print("=================================================================")
    print(f" Input Path    : {input_path}")
    print(f" Config File   : {args.config}")
    print(f" Selected Mode : {args.mode}")
    print("=================================================================\n")

    # Run Aggregator
    aggregator = ResultsAggregator(config)
    summary_df = aggregator.run(input_path, mode=args.mode)

    # Optional Evaluation Mode
    gt_csv = args.evaluate if args.evaluate else config.get("paths", {}).get("ground_truth_csv", "")
    if gt_csv and os.path.exists(gt_csv):
        summary_csv_path = os.path.join(
            config.get("paths", {}).get("output_dir", "output_results"), "summary.csv"
        )
        evaluate_predictions(summary_csv_path, gt_csv)


if __name__ == "__main__":
    main()
