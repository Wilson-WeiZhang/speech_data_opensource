#!/usr/bin/env python3
"""Summarize 58-participant FAST benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def subject_scores(epoacc: np.ndarray) -> np.ndarray:
    if epoacc.ndim == 3:
        return epoacc[:, -1, :].mean(axis=1)
    if epoacc.ndim == 2:
        return epoacc[:, -1]
    raise ValueError(f"Unsupported epoacc shape: {epoacc.shape}")


def row_from_values(method: str, values: np.ndarray, artifact: Path | str) -> dict:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError(f"{method}: no values")
    return {
        "method": method,
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "artifact": str(artifact),
    }


def summarize_fast(run_dir: Path, kind: str, method: str) -> dict:
    preferred = run_dir / f"{kind}_58.npz"
    fallback = run_dir / f"{kind}.npz"
    path = preferred if preferred.exists() else fallback
    if not path.exists():
        raise FileNotFoundError(f"Missing {preferred} or {fallback}")
    data = np.load(path, allow_pickle=True)
    return row_from_values(method, subject_scores(data["epoacc"]), path)


def summarize_split_dir(directory: Path, method: str) -> dict:
    paths = sorted(directory.glob("fold*.npz")) + sorted(directory.glob("split*.npz"))
    values: list[float] = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        key = "subject_mean" if "subject_mean" in data else "subject_bacc" if "subject_bacc" in data else None
        if key is None:
            continue
        values.extend(float(x) for x in np.asarray(data[key]).ravel())
    if not values:
        raise ValueError(f"{method}: no fold/split values found in {directory}")
    return row_from_values(method, np.array(values, dtype=float), directory)


def write_tsv(rows: list[dict], data_version: str, out_tsv: Path) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        f.write("data_version\tmethod\tn\tmean\tsd\tmin\tmax\tartifact\n")
        for row in rows:
            f.write(
                f"{data_version}\t{row['method']}\t{row['n']}\t"
                f"{row['mean']:.6f}\t{row['sd']:.6f}\t{row['min']:.6f}\t"
                f"{row['max']:.6f}\t{row['artifact']}\n"
            )


def validate_expected_n(rows: list[dict], expected_n: int) -> None:
    bad = [f"{row['method']} n={row['n']}" for row in rows if row["n"] != expected_n]
    if bad:
        raise ValueError(f"Expected n={expected_n} for every method, got {', '.join(bad)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fast-run-dir", required=True, type=Path)
    p.add_argument("--svm-dir", required=True, type=Path)
    p.add_argument("--eegnet-dir", required=True, type=Path)
    p.add_argument("--out-tsv", required=True, type=Path)
    p.add_argument("--data-version", default="fast_benchmark_58")
    p.add_argument("--expected-n", default=58, type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        summarize_fast(args.fast_run_dir, "pre", "FAST-pre"),
        summarize_fast(args.fast_run_dir, "tune", "FAST-tune"),
        summarize_split_dir(args.svm_dir, "SVM"),
        summarize_split_dir(args.eegnet_dir, "EEGNet"),
    ]
    validate_expected_n(rows, args.expected_n)
    write_tsv(rows, args.data_version, args.out_tsv)
    print(f"Wrote {args.out_tsv}")


if __name__ == "__main__":
    main()
