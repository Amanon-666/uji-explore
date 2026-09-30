"""Check saved graph experiment hashes and recompute reported Euclidean errors."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("graph_runs"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/graph_run_hashes.json"))
    args = parser.parse_args()
    for name, hashes in json.loads(args.manifest.read_text()).items():
        directory = args.runs / name
        checks = {"FROZEN.json": hashes["frozen_sha256"],
                  "metrics.csv": hashes["metrics_sha256"],
                  "predictions.csv": hashes["predictions_sha256"],
                  **hashes["checkpoint_sha256"]}
        for relative, expected in checks.items():
            file = directory / relative
            if sha(file) != expected:
                raise ValueError(f"Hash mismatch: {file}")
        predictions = pd.read_csv(directory / "predictions.csv")
        metrics = pd.read_csv(directory / "metrics.csv")
        for role, subset in predictions.groupby("role"):
            groups = subset.groupby("method") if "method" in subset else [("GConvLoc-reconstruction", subset)]
            for method, rows in groups:
                delta = rows[["pred_x", "pred_y"]].to_numpy() - rows[["LONGITUDE", "LATITUDE"]].to_numpy()
                mean_error = np.linalg.norm(delta, axis=1).mean()
                recorded = metrics[(metrics.role == role) & (metrics.method == method)]
                if "building" in recorded:
                    recorded = recorded[recorded.building == "all"]
                if len(recorded) != 1 or abs(mean_error - recorded.mde_m.iloc[0]) > 1e-5:
                    raise ValueError(f"Metric mismatch: {name}/{role}/{method}")
        print(f"PASS {name}: hashes and mean Euclidean errors")


if __name__ == "__main__":
    main()
