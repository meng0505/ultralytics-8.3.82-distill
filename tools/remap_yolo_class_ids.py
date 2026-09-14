#!/usr/bin/env python3
"""Create an audited YOLO-label copy with explicit contiguous class IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

CLASS_LINE = re.compile(r"^(\s*)(\d+)(\s+.*)?$")


def parse_mapping(values: list[str]) -> dict[int, int]:
    """Parse OLD:NEW class mapping tokens and reject ambiguous mappings."""
    mapping: dict[int, int] = {}
    for value in values:
        try:
            old_text, new_text = value.split(":", 1)
            old, new = int(old_text), int(new_text)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid mapping {value!r}; expected OLD:NEW") from error
        if old < 0 or new < 0:
            raise ValueError("class IDs must be non-negative")
        if old in mapping and mapping[old] != new:
            raise ValueError(f"source class {old} has multiple targets")
        mapping[old] = new
    if not mapping:
        raise ValueError("at least one --map OLD:NEW entry is required")
    targets = sorted(set(mapping.values()))
    if targets != list(range(len(targets))):
        raise ValueError(f"target IDs must be contiguous from zero, got {targets}")
    return mapping


def remap_tree(source: Path, output: Path, mapping: dict[int, int]) -> dict:
    """Remap every TXT label atomically while retaining all geometry text."""
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source label directory does not exist: {source}")
    if source == output or source in output.parents:
        raise ValueError("output must not be the source directory or one of its children")

    label_files = sorted(source.rglob("*.txt"))
    if not label_files:
        raise ValueError(f"no TXT labels found under {source}")

    class_counts = {str(target): 0 for target in sorted(set(mapping.values()))}
    digest = hashlib.sha256()
    line_count = 0
    for source_file in label_files:
        relative = source_file.relative_to(source)
        target_file = output / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        remapped_lines = []
        for line_number, line in enumerate(source_file.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                remapped_lines.append("")
                continue
            match = CLASS_LINE.match(line)
            if match is None:
                raise ValueError(f"{source_file}:{line_number}: invalid YOLO label line")
            source_class = int(match.group(2))
            if source_class not in mapping:
                raise ValueError(
                    f"{source_file}:{line_number}: unmapped source class {source_class}; "
                    f"allowed={sorted(mapping)}"
                )
            target_class = mapping[source_class]
            remapped_lines.append(f"{match.group(1)}{target_class}{match.group(3) or ''}")
            class_counts[str(target_class)] += 1
            line_count += 1
        text = "\n".join(remapped_lines) + ("\n" if remapped_lines else "")
        temporary = target_file.with_suffix(target_file.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target_file)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))

    expected = {path.relative_to(source) for path in label_files}
    unexpected = [
        path
        for path in output.rglob("*.txt")
        if path.relative_to(output) not in expected
    ]
    if unexpected:
        raise ValueError(f"output contains unexpected label files: {unexpected[:8]}")

    manifest = {
        "schema_version": 1,
        "source": str(source),
        "output": str(output),
        "mapping": {str(old): new for old, new in sorted(mapping.items())},
        "label_files": len(label_files),
        "instances": line_count,
        "class_counts": class_counts,
        "content_sha256": digest.hexdigest(),
    }
    manifest_path = output / "remap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    """Parse command line and write the audited label copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map", nargs="+", required=True, dest="mapping")
    args = parser.parse_args()
    manifest = remap_tree(args.source, args.output, parse_mapping(args.mapping))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
