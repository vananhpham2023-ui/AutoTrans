#!/usr/bin/env python3
"""Aggregate PINN training CSV files and emit normalizer statistics.

The script scans one or multiple `pinn_dataset_*.csv` files, computes the
per-dimension min/max/mean/std for the 13 network inputs and 6 outputs, and
writes the consolidated results to JSON (and optionally pickle) files. These
statistics are consumed by the C++ PINN estimator to reproduce the exact
MinMax scaling that was used during training.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


INPUT_FIELDS = [
    "quad_acc_x",
    "quad_acc_y",
    "quad_acc_z",
    "load_acc_x",
    "load_acc_y",
    "load_acc_z",
    "cable_dir_x",
    "cable_dir_y",
    "cable_dir_z",
    "quad_rot_r13",
    "quad_rot_r23",
    "quad_rot_r33",
    "thrust_total",
]

OUTPUT_FIELDS = [
    "fq_true_x",
    "fq_true_y",
    "fq_true_z",
    "fl_true_x",
    "fl_true_y",
    "fl_true_z",
]

FIELD_ALIASES = {
    # Input aliases originating from legacy recorder scripts.
    "quad_acc_x": ["accQ_x"],
    "quad_acc_y": ["accQ_y"],
    "quad_acc_z": ["accQ_z"],
    "load_acc_x": ["accL_x"],
    "load_acc_y": ["accL_y"],
    "load_acc_z": ["accL_z"],
    "cable_dir_x": ["rho_x"],
    "cable_dir_y": ["rho_y"],
    "cable_dir_z": ["rho_z"],
    "quad_rot_r13": ["ez_world_x"],
    "quad_rot_r23": ["ez_world_y"],
    "quad_rot_r33": ["ez_world_z"],
    "thrust_total": ["thrust"],
    # Output aliases.
    "fq_true_x": ["fQ_x"],
    "fq_true_y": ["fQ_y"],
    "fq_true_z": ["fQ_z"],
    "fl_true_x": ["fL_x"],
    "fl_true_y": ["fL_y"],
    "fl_true_z": ["fL_z"],
}


@dataclass
class DimStats:
    name: str
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = field(default_factory=lambda: math.inf)
    maximum: float = field(default_factory=lambda: -math.inf)

    def update(self, value: float) -> None:
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Encountered non-finite value for field '{self.name}'")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def serialize(self) -> Dict[str, float]:
        return {
            "min": self.minimum if self.count else 0.0,
            "max": self.maximum if self.count else 0.0,
            "mean": self.mean if self.count else 0.0,
            "std": self.std if self.count else 0.0,
        }


def _build_stats(fields: Sequence[str]) -> Dict[str, DimStats]:
    return {name: DimStats(name=name) for name in fields}


def _resolve_files(patterns: Sequence[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(set(files))


def _resolve_source_key(row: Dict[str, str], canonical: str) -> str | None:
    if canonical in row:
        return canonical
    for alias in FIELD_ALIASES.get(canonical, []):
        if alias in row:
            return alias
    return None


def _load_row(row: Dict[str, str], stats: Dict[str, DimStats]) -> None:
    for key, dim in stats.items():
        source_key = _resolve_source_key(row, key)
        if source_key is None:
            aliases = FIELD_ALIASES.get(key)
            if aliases:
                formatted_aliases = ", ".join(aliases)
                raise KeyError(f"Column '{key}' (aliases: {formatted_aliases}) is missing from the dataset")
            raise KeyError(f"Column '{key}' is missing from the dataset")
        dim.update(float(row[source_key]))


def compute_stats(files: Sequence[str]) -> Dict[str, Dict[str, List[float]]]:
    input_stats = _build_stats(INPUT_FIELDS)
    output_stats = _build_stats(OUTPUT_FIELDS)
    sample_count = 0
    skipped_samples = 0

    for path in files:
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    _load_row(row, input_stats)
                    _load_row(row, output_stats)
                except ValueError:
                    skipped_samples += 1
                    continue
                else:
                    sample_count += 1

    if not sample_count:
        raise RuntimeError("No samples found in the provided dataset files")

    def pack(collection: Dict[str, DimStats]) -> Dict[str, List[float]]:
        ordered = [collection[name].serialize() for name in collection]
        return {
            "fields": list(collection.keys()),
            "min": [item["min"] for item in ordered],
            "max": [item["max"] for item in ordered],
            "mean": [item["mean"] for item in ordered],
            "std": [item["std"] for item in ordered],
        }

    return {
        "summary": {
            "files": list(files),
            "samples": sample_count,
            "skipped_samples": skipped_samples,
        },
        "input": pack(input_stats),
        "output": pack(output_stats),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="*",
        default=["plots/pinn_dataset/pinn_dataset_*.csv"],
        help="Glob patterns pointing to pinn_dataset CSV files",
    )
    parser.add_argument(
        "--output",
        default="models/normalizer_stats.json",
        help="Destination JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--pickle",
        default=None,
        help="Optional path to also store a pickle version of the stats",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation",
    )
    args = parser.parse_args(argv)

    files = _resolve_files(args.sources)
    if not files:
        raise SystemExit("No dataset files matched the provided patterns")

    stats = compute_stats(files)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2 if args.pretty else None, sort_keys=False)
        handle.write("\n")

    if args.pickle:
        os.makedirs(os.path.dirname(os.path.abspath(args.pickle)), exist_ok=True)
        with open(args.pickle, "wb") as handle:
            pickle.dump(stats, handle)

    print(f"Wrote normalizer statistics to {args.output}")
    if args.pickle:
        print(f"Wrote pickle artifact to {args.pickle}")


if __name__ == "__main__":
    main()
