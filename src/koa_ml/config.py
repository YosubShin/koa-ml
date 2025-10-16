from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml


DEFAULT_CONFIG_PATH = Path("~/.config/koa-ml/config.yaml").expanduser()


@dataclass
class Config:
    user: str
    host: str
    identity_file: Optional[Path] = None
    remote_workdir: Path = Path("~/koa-ml")
    proxy_command: Optional[str] = None

    @property
    def login(self) -> str:
        return f"{self.user}@{self.host}"


PathLikeOrStr = Union[os.PathLike[str], str]


def load_config(config_path: Optional[PathLikeOrStr] = None) -> Config:
    """
    Load configuration from disk. When no path is provided we fall back to
    ~/.config/koa-ml/config.yaml and merge it with environment overrides.
    """

    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {path}. "
            "Run `cp .koa-config.example.yaml ~/.config/koa-ml/config.yaml` "
            "and update the credentials."
        )

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    env_overrides = {
        "user": os.getenv("KOA_USER"),
        "host": os.getenv("KOA_HOST"),
        "identity_file": os.getenv("KOA_IDENTITY_FILE"),
        "remote_workdir": os.getenv("KOA_REMOTE_WORKDIR"),
        "proxy_command": os.getenv("KOA_PROXY_COMMAND"),
    }

    for key, value in env_overrides.items():
        if value is not None:
            data[key] = value

    missing = [key for key in ("user", "host") if not data.get(key)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    identity_file = data.get("identity_file") or None
    identity_path: Optional[Path] = None
    if identity_file:
        identity_path = Path(identity_file).expanduser()
        if not identity_path.exists():
            raise FileNotFoundError(
                f"Configured identity_file not found: {identity_path}. "
                "Update the path or remove the identity_file setting to rely on your SSH defaults."
            )

    return Config(
        user=data["user"],
        host=data["host"],
        identity_file=identity_path,
        remote_workdir=Path(data.get("remote_workdir", "~/koa-ml")),
        proxy_command=data.get("proxy_command") or None,
    )
