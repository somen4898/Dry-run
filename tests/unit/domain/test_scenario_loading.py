"""Tests for YAML scenario loading."""

import pytest
from pathlib import Path
import yaml
from dryrun.domain.models.scenario import Scenario


HAPPY_PATH_YAML = """\
id: happy-path-001
name: "Happy path - laptop purchase"
description: "User successfully purchases a laptop through the support agent"
persona:
  goal: "Buy a budget laptop for college"
  tone: "polite"
  knowledge_level: "novice"
  background: "College freshman, first time buying a laptop online"
  goal_reveal_strategy: "incremental"
opening_input: "Hi, I need help finding a laptop"
expectations:
  required_tools:
    - search_inventory
    - add_to_cart
    - process_checkout
  required_tool_args:
    search_inventory:
      query: "laptop"
  terminal_state: null
  output_must_contain:
    - "order"
constraints:
  max_turns: 8
  timeout_seconds: 120
  max_tokens: 8000
golden: true
tags:
  - smoke
  - purchase
"""


class TestScenarioFromYAML:
    def test_load_from_yaml_string(self):
        data = yaml.safe_load(HAPPY_PATH_YAML)
        scenario = Scenario(**data)
        assert scenario.id == "happy-path-001"
        assert scenario.persona.goal_reveal_strategy == "incremental"
        assert "search_inventory" in scenario.expectations.required_tools
        assert scenario.golden is True

    def test_load_from_yaml_file(self, tmp_path: Path):
        p = tmp_path / "test_scenario.yaml"
        p.write_text(HAPPY_PATH_YAML)
        data = yaml.safe_load(p.read_text())
        scenario = Scenario(**data)
        assert scenario.name == "Happy path - laptop purchase"

    def test_missing_required_field_raises(self):
        bad_yaml = "id: test\nname: test\n"
        data = yaml.safe_load(bad_yaml)
        with pytest.raises(Exception):
            Scenario(**data)

    def test_default_constraints(self):
        data = yaml.safe_load(HAPPY_PATH_YAML)
        data["constraints"] = {}
        scenario = Scenario(**data)
        assert scenario.constraints.max_turns == 10
