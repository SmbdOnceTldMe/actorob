from pathlib import Path
import pytest

from actorob.mjcf import resolve_mjcf_path
from actorob.models import copy_mjcf_with_resolved_assets


def test_resolve_mjcf_path_returns_existing_file():
    root = Path(__file__).resolve().parents[1]
    mjcf_path = root / "robots" / "dog" / "dog.xml"

    assert resolve_mjcf_path(mjcf_path) == mjcf_path.resolve()


def test_resolve_mjcf_path_does_not_fall_back_to_robot_xml_alias(tmp_path: Path):
    robot_dir = tmp_path / "dog"
    robot_dir.mkdir()
    alias_path = robot_dir / "dog.xml"
    alias_path.write_text("<mujoco/>", encoding="utf-8")
    requested_path = robot_dir / "robot.xml"

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_mjcf_path(requested_path)

    assert str(requested_path) in str(exc_info.value)
    assert alias_path.exists()


def test_copy_mjcf_with_resolved_assets_rewrites_relative_compiler_dirs(tmp_path: Path):
    source_dir = tmp_path / "robot"
    source_dir.mkdir()
    source = source_dir / "robot.xml"
    source.write_text(
        '<mujoco model="x"><compiler meshdir="meshes/" texturedir="textures/"/></mujoco>',
        encoding="utf-8",
    )

    target = tmp_path / "outputs" / "best.xml"
    saved = copy_mjcf_with_resolved_assets(source, target)
    content = saved.read_text(encoding="utf-8")

    assert str((source_dir / "meshes").resolve()) in content
    assert str((source_dir / "textures").resolve()) in content
