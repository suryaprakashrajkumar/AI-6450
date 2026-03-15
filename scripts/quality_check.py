from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.dataset import run_quality_checks, write_qa_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--fail_on_warning", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    reports_dir = data_dir / "qa_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": data_dir / "train_samples.npz",
        "val": data_dir / "val_samples.npz",
        "test": data_dir / "test_samples.npz",
    }

    any_error = False
    any_warning = False

    for split, path in split_files.items():
        report = run_quality_checks(str(path), config.qa_action_imbalance_ratio_threshold)
        out_path = reports_dir / f"{split}_qa.txt"
        write_qa_report(report, str(out_path))
        print(f"[{split}] passed={report.passed} errors={len(report.errors)} warnings={len(report.warnings)}")

        any_error = any_error or (not report.passed)
        any_warning = any_warning or (len(report.warnings) > 0)

    if any_error:
        raise SystemExit("Dataset QA failed due to errors.")
    if args.fail_on_warning and any_warning:
        raise SystemExit("Dataset QA failed due to warnings (strict mode).")


if __name__ == "__main__":
    main()
