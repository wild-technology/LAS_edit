"""Application preferences and settings."""
from pathlib import Path
from PySide6.QtCore import QSettings


_ORG = "WildTechnologies"
_APP = "PointCloudEditor"


def get_settings() -> QSettings:
    return QSettings(_ORG, _APP)


def get_recent_projects() -> list[str]:
    s = get_settings()
    return s.value("recent_projects", []) or []


def add_recent_project(path: str):
    s = get_settings()
    recent = get_recent_projects()
    path = str(Path(path).resolve())
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    s.setValue("recent_projects", recent[:10])


def get_viewport_point_budget() -> int:
    return int(get_settings().value("viewport_point_budget", 10_000_000))


def set_viewport_point_budget(budget: int):
    get_settings().setValue("viewport_point_budget", budget)


def get_lod_max_depth() -> int:
    return int(get_settings().value("lod_max_depth", 8))


def get_lod_cache_dir() -> Path:
    default = Path.home() / ".pointcloud_editor_cache"
    path = get_settings().value("lod_cache_dir", str(default))
    return Path(path)
