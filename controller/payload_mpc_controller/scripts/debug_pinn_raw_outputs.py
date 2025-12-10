#!/usr/bin/env python3
"""
Replay samples from a PINN dataset/log, rebuild the 13-D input feature vector,
run the TorchScript estimator, and dump both the normalized inputs and the
network's raw outputs (before any denormalization or saturation).

Usage example:
  rosrun payload_mpc_controller debug_pinn_raw_outputs.py \
      --csv plots/pinn_dataset/pinn_dataset_train.csv \
      --model models/force_estimation_script3.pt \
      --normalizer models/normalizer_stats.json \
      --limit 1000 \
      --output /tmp/pinn_raw_dump.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import torch

CANONICAL_INPUT_FIELDS: Sequence[str] = (
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
)

ROW_ALIASES: Dict[str, Sequence[str]] = {
    "quad_acc_x": ("accQ_x",),
    "quad_acc_y": ("accQ_y",),
    "quad_acc_z": ("accQ_z",),
    "load_acc_x": ("accL_x",),
    "load_acc_y": ("accL_y",),
    "load_acc_z": ("accL_z",),
    "cable_dir_x": ("rho_x",),
    "cable_dir_y": ("rho_y",),
    "cable_dir_z": ("rho_z",),
    "quad_rot_r13": ("ez_world_x",),
    "quad_rot_r23": ("ez_world_y",),
    "quad_rot_r33": ("ez_world_z",),
    "thrust_total": ("thrust",),
}

OUTPUT_FIELDS: Sequence[str] = (
    "fq_x_norm",
    "fq_y_norm",
    "fq_z_norm",
    "fl_x_norm",
    "fl_y_norm",
    "fl_z_norm",
)


def _resolve_column(row: Dict[str, str], canonical: str) -> float:
    if canonical in row:
        return float(row[canonical])
    for alias in ROW_ALIASES.get(canonical, ()):
        if alias in row:
            return float(row[alias])
    raise KeyError(f"Missing '{canonical}' (aliases={ROW_ALIASES.get(canonical)}) in row header.")


def _normalize(value: float, min_v: float, max_v: float) -> float:
    span = max_v - min_v
    if span <= 1e-9:
        return 0.0
    clamped = min(max(value, min_v), max_v)
    return 2.0 * (clamped - min_v) / span - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV file containing feature columns.")
    parser.add_argument("--model", required=True, help="TorchScript .pt model path.")
    parser.add_argument("--normalizer", required=True, help="Normalizer JSON path.")
    parser.add_argument("--output", default="/tmp/pinn_raw_dump.csv", help="Where to store the dump CSV.")
    parser.add_argument("--limit", type=int, default=2000, help="Max number of samples to process.")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    normalizer_path = Path(args.normalizer).expanduser().resolve()

    with normalizer_path.open("r", encoding="utf-8") as handle:
        normalizer = json.load(handle)
    input_stats = dict(zip(CANONICAL_INPUT_FIELDS, zip(normalizer["input"]["min"], normalizer["input"]["max"])))

    module = torch.jit.load(str(model_path))
    module.eval()

    processed = 0
    raw_outputs: List[List[float]] = []

    with csv_path.open("r", newline="") as csv_handle, Path(args.output).open("w", newline="") as dump_handle:
        reader = csv.DictReader(csv_handle)
        writer = csv.writer(dump_handle)
        writer.writerow(["sample_index"] + list(CANONICAL_INPUT_FIELDS) + list(OUTPUT_FIELDS))

        for row in reader:
            features = []
            normalized = []
            for field in CANONICAL_INPUT_FIELDS:
                value = _resolve_column(row, field)
                features.append(value)
                min_v, max_v = input_stats[field]
                normalized.append(_normalize(value, min_v, max_v))

            tensor = torch.tensor([normalized], dtype=torch.float64)
            with torch.no_grad():
                raw = module(tensor).reshape(-1).cpu().tolist()
            writer.writerow([processed] + features + raw)
            if all(map(math.isfinite, raw)):
                raw_outputs.append(raw)

            processed += 1
            if processed >= args.limit:
                break

    if not raw_outputs:
        print("No samples processed.")
        return

    import statistics

    print(f"Processed {processed} samples. Dump written to {args.output}")
    for idx, name in enumerate(OUTPUT_FIELDS):
        values = [row[idx] for row in raw_outputs if math.isfinite(row[idx])]
        if not values:
            print(f"{name}: no finite samples (likely NaN from the model).")
            continue
        mean_val = statistics.mean(values)
        std_val = statistics.pstdev(values)
        print(
            f"{name}: min={min(values):.4f} max={max(values):.4f} "
            f"mean={mean_val:.4f} std={std_val:.4f}"
        )


if __name__ == "__main__":
    main()
