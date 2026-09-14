#!/usr/bin/env python3
"""Validate optical/SAR/label identity and image dimensions without positional pairing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.data.paired_obb_dataset import build_pair_records  # noqa: E402


def main() -> None:
    """Run the pair validator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sar", required=True)
    parser.add_argument("--optical", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--manifest")
    parser.add_argument("--split")
    parser.add_argument("--filename-mapper")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    records, pairing = build_pair_records(
        args.sar,
        args.optical,
        args.labels,
        manifest=args.manifest,
        split=args.split,
        sample_id_fn=args.filename_mapper,
        require_labels=args.labels is not None,
    )
    shape_mismatches, corrupt = [], []
    for record in records:
        sar = cv2.imread(record.sar_path)
        optical = cv2.imread(record.optical_path)
        if sar is None or optical is None:
            corrupt.append(record.sample_id)
        elif sar.shape[:2] != optical.shape[:2]:
            shape_mismatches.append(
                {"sample_id": record.sample_id, "sar": list(sar.shape[:2]), "optical": list(optical.shape[:2])}
            )
    result = {
        "pairing": pairing.to_dict(),
        "corrupt_ids": corrupt,
        "shape_mismatches": shape_mismatches,
        "valid": not pairing.has_errors and not corrupt and not shape_mismatches,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if args.strict and not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
