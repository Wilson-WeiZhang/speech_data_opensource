#!/usr/bin/env python3
"""Compatibility wrapper for standalone FAST benchmark preprocessing."""

from __future__ import annotations

import sys

from benchmark_eeg_preprocess import main


if "--dataset" not in sys.argv:
    sys.argv[1:1] = ["--dataset", "st"]

if __name__ == "__main__":
    main()
