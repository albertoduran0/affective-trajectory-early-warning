"""Output path resolution helpers protecting precomputed results by default."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_overwrite_flag(argv: list[str] | None = None) -> bool:
    """Return True when --overwrite is explicitly passed on the CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite precomputed outputs in results/ instead of writing to results/rerun/.",
    )
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return bool(args.overwrite)


def get_output_dirs(root: Path, overwrite: bool | None = None) -> tuple[Path, Path]:
    """Return (tables_dir, figures_dir), writing to results/rerun/ unless overwrite=True."""
    if overwrite is None:
        overwrite = parse_overwrite_flag()
    base = root / "results" if overwrite else root / "results" / "rerun"
    tables_dir = base / "tables"
    figures_dir = base / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def resolve_input_table(root: Path, filename: str, prefer_rerun: bool = False) -> Path:
    """Resolve an input table path, preferring results/rerun/tables when present."""
    if prefer_rerun:
        rerun_path = root / "results" / "rerun" / "tables" / filename
        if rerun_path.exists():
            return rerun_path
    return root / "results" / "tables" / filename

