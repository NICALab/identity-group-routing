from __future__ import annotations

from pathlib import Path


def snapshot_clean_pack_entrypoint(
    *,
    run_dir: Path,
    argv: list[str],
    entry_script: Path,
    extra_files: list[Path] | None = None,
) -> None:
    return None
