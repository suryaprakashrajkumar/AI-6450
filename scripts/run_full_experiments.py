from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_large.json")
    parser.add_argument("--models", type=str, default="mlp,cnn_small,cnn_large")
    parser.add_argument("--out_dir", type=str, default="logs/ablation")
    args = parser.parse_args()

    py = sys.executable

    _run([py, "-m", "scripts.generate_data", "--config", args.config])
    _run([py, "-m", "scripts.quality_check", "--config", args.config])
    _run(
        [
            py,
            "-m",
            "scripts.ablation_study",
            "--config",
            args.config,
            "--models",
            args.models,
            "--out_dir",
            args.out_dir,
            "--include_sklearn",
        ]
    )
    _run([py, "-m", "scripts.plot_ablation", "--input", f"{args.out_dir}/ablation_results.csv", "--out_dir", f"{args.out_dir}/charts"])

    print("Full experiment pipeline complete.")


if __name__ == "__main__":
    main()
