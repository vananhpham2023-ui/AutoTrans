#!/usr/bin/env python3
"""
Merge multiple `pinn_dataset_*.csv` files into a single clean dataset.

用途：
- 将 `batch_data_collection.py` 生成的 run_* 场景 CSV，
  以及 `collect_figure_eight_dataset.py` 生成的额外 figure-eight
  场景 CSV，统一合并为一个大训练集供 PINN 使用。

使用方法（在 catkin 工作区根目录）：

    python3 src/AutoTrans/controller/payload_mpc_controller/scripts/merge_pinn_datasets.py \\
        --roots src/AutoTrans/controller/payload_mpc_controller/plots \\
        --output src/AutoTrans/controller/payload_mpc_controller/plots/pinn_dataset/clean_dataset_task9_v2.csv

脚本会从 `--roots` 指定的目录递归搜索所有 `pinn_dataset_*.csv`，
按统一表头合并，并做一次基本的数值有效性检查（跳过含 NaN/Inf 或
推力超出 [0,150] N 的行）。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List


REQUIRED_FIELDS = [
    "timestamp",
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
    "fl_true_x",
    "fl_true_y",
    "fl_true_z",
    "fq_true_x",
    "fq_true_y",
    "fq_true_z",
]


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="One or more root directories to search for pinn_dataset_*.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for merged dataset",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _is_valid_row(row: dict) -> bool:
    try:
        for key in REQUIRED_FIELDS:
            if key not in row or row[key].strip() == "":
                return False
        values = [float(row[key]) for key in REQUIRED_FIELDS if key != "timestamp"]
        if not all(math.isfinite(v) for v in values):
            return False
        thrust = abs(float(row.get("thrust_total", "0.0")))
        if thrust <= 0.0 or thrust > 150.0:
            return False
        return True
    except ValueError:
        return False


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    roots = [Path(r).resolve() for r in args.roots]
    output_path = Path(args.output).resolve()

    csv_paths: List[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("pinn_dataset_*.csv"):
            if path not in seen:
                seen.add(path)
                csv_paths.append(path)

    if not csv_paths:
        raise SystemExit("[merge_pinn_datasets] 未在指定 roots 中找到任何 pinn_dataset_*.csv 文件。")

    csv_paths.sort()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_written = False
    kept = 0
    with output_path.open("w", newline="") as out_file:
        writer: csv.writer = csv.writer(out_file)
        header: List[str] = []
        for path in csv_paths:
            with path.open() as in_file:
                reader = csv.DictReader(in_file)
                if not reader.fieldnames:
                    continue
                if not header_written:
                    header = list(reader.fieldnames)
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    if not row:
                        continue
                    if _is_valid_row(row):
                        writer.writerow([row.get(col, "") for col in header])
                        kept += 1

    print(
        f"[merge_pinn_datasets] 合并 {len(csv_paths)} 个 CSV，"
        f"最终有效样本 {kept} 行 → {output_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()

