"""Load a YAML config, resolving an optional `base:` file it inherits from."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base:
        merged = load_config(path.parent / base)
        merged.update(cfg)
        cfg = merged
    return cfg


def resolve(p):
    """Resolve a config path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
