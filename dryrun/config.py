"""DryRunConfig — Pydantic config loaded from dryrun.yaml."""
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
import yaml


class ModelConfig(BaseModel):
    synthetic_user: str = "gpt-4o-mini"


class DryRunConfig(BaseModel):
    agent_module: str
    agent_object: str
    scenarios_dir: str = "scenarios/"
    models: ModelConfig = ModelConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> DryRunConfig:
        data = yaml.safe_load(path.read_text())
        return cls(**data)
