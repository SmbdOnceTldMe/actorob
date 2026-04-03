"""Helpers for copying MJCF files together with resolved asset directories."""

from __future__ import annotations

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def copy_mjcf_with_resolved_assets(source_path: str | Path, target_path: str | Path) -> Path:
    """Copy an MJCF file and rewrite relative asset directories to absolute ones."""

    source = Path(source_path).expanduser().resolve()
    target = Path(target_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    tree = ET.parse(target)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        _rewrite_compiler_dir(compiler, "meshdir", source.parent)
        _rewrite_compiler_dir(compiler, "texturedir", source.parent)
        tree.write(target, encoding="utf-8")

    return target


def _rewrite_compiler_dir(compiler: ET.Element, attr_name: str, source_dir: Path) -> None:
    raw_value = compiler.attrib.get(attr_name)
    if raw_value is None or raw_value.strip() == "":
        return

    raw_path = Path(raw_value)
    if raw_path.is_absolute():
        return

    resolved = (source_dir / raw_path).resolve()
    compiler.set(attr_name, str(resolved))


__all__ = ["copy_mjcf_with_resolved_assets"]
