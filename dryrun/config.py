"""DryRunConfig — Pydantic config loaded from dryrun.yaml."""

from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
import yaml


class ModelConfig(BaseModel):
    provider: Literal["openai", "anthropic"] = "anthropic"
    synthetic_user: str = "claude-sonnet-4-6"
    agent: str = "claude-sonnet-4-6"


class DryRunConfig(BaseModel):
    agent_module: str
    agent_object: str
    scenarios_dir: str = "scenarios/"
    models: ModelConfig = ModelConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> DryRunConfig:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Invalid config file: {path} (expected a YAML mapping)")
        return cls(**data)
