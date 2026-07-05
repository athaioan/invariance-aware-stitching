#!/usr/bin/env python3
"""
Recursively finds YAML files named FuLA_<n>.yaml (or .yml) in the current
directory and all subdirectories, and renames them to IRIS_<n>.yaml (.yml).

Usage:
    python rename_fula_to_iris.py [root_dir] [--dry-run]

If root_dir is omitted, the current directory is used.
Use --dry-run to preview the renames without actually changing anything.
"""

import argparse
import re
from pathlib import Path

# Matches: FuLA_<number>.yaml or FuLA_<number>.yml (case-sensitive on "FuLA")
PATTERN = re.compile(r"^FuLA_(\d+)\.(yaml|yml)$")


def find_and_rename(root: Path, dry_run: bool = False):
    matches = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))

    renamed = 0
    for path in matches:
        m = PATTERN.match(path.name)
        if not m:
            continue

        number, ext = m.groups()
        new_name = f"IRIS_{number}.{ext}"
        new_path = path.with_name(new_name)

        if new_path.exists():
            print(f"SKIP  (target exists): {path} -> {new_path}")
            continue

        if dry_run:
            print(f"DRY-RUN: {path} -> {new_path}")
        else:
            path.rename(new_path)
            print(f"RENAMED: {path} -> {new_path}")
        renamed += 1

    if renamed == 0:
        print("No matching files found (pattern: FuLA_<number>.yaml/.yml).")
    else:
        action = "would be renamed" if dry_run else "renamed"
        print(f"\n{renamed} file(s) {action}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=".",
        help="Root directory to search (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually renaming",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"Error: {root} is not a valid directory")

    find_and_rename(root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()