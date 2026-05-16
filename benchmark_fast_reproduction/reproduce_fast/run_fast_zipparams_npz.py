from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def patch_train_fast_aggregation(train_fast: Path, n_subjects: int) -> bool:
    text = train_fast.read_text(encoding="utf-8")
    patched_lines = []
    changed = False
    for line in text.splitlines(keepends=True):
        is_aggregation = "convert_epoch_acc_to_npy" in line or "convert_logits_to_npy" in line
        if is_aggregation and "range(57)" in line:
            line = line.replace("range(57)", f"range({n_subjects})")
            changed = True
        patched_lines.append(line)
    if changed:
        train_fast.write_text("".join(patched_lines), encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/media/datasets/DSO_SD/reproduce_FAST")
    parser.add_argument("--import-dir", required=True)
    parser.add_argument("--run-cwd", required=True)
    parser.add_argument("--train-fast", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--bs", default="200")
    parser.add_argument("--n-subjects", type=int, default=58)
    args = parser.parse_args()

    run_cwd = Path(args.run_cwd)
    run_cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(run_cwd)

    patch_train_fast_aggregation(Path(args.train_fast), args.n_subjects)

    os.environ["COSMO_BENCHMARK_NPZ"] = args.npz
    # Compatibility for the original FAST data loader imported at runtime.
    os.environ["UP2025_58_NPZ"] = args.npz
    sys.path.insert(0, args.import_dir)
    sys.path.insert(1, args.repo_root)

    # Match the original reproduce_FAST.zip train_FAST defaults explicitly.
    sys.argv = [
        "train_FAST.py",
        "--gpu",
        args.gpu,
        "--ds",
        args.dataset_name,
        "--folds",
        args.folds,
        "--utr",
        "12",
        "--zone",
        "A",
        "--dim1",
        "64",
        "--dim2",
        "96",
        "--head",
        "V0",
        "--win",
        "200",
        "--step",
        "200",
        "--lay",
        "4",
        "--bs",
        args.bs,
        "--seed",
        "42",
    ]
    runpy.run_path(args.train_fast, run_name="__main__")


if __name__ == "__main__":
    main()
