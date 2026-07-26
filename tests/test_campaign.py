import pytest

from pvzwm.campaign import (
    campaign_stages,
    campaign_status,
    port_scenarios,
    supported_campaign_stage_ids,
)
from pvzwm.campaign_proxy import build_campaign_proxy
from pvzwm.python_port import PythonPortConfig


def test_campaign_manifest_uses_sequential_stage_ids() -> None:
    stages = campaign_stages()
    assert len(stages) == 50
    assert stages[0].stage_id == "1-1"
    assert stages[-1].stage_id == "5-10"
    assert [stage.stage_id for stage in stages[9:12]] == ["1-10", "2-1", "2-2"]
    assert next(stage for stage in stages if stage.stage_id == "3-1").rows == 6
    assert next(stage for stage in stages if stage.stage_id == "4-5").area == "night"
    assert next(stage for stage in stages if stage.stage_id == "5-10").mode == "boss"
    assert stages[0].implementation == "partial"
    assert stages[1].implementation == "partial"
    assert stages[2].implementation == "proxy"
    assert stages[4].implementation == "missing"
    assert next(stage for stage in stages if stage.stage_id == "2-1").implementation == "proxy"


def test_port_demos_are_not_campaign_stages() -> None:
    scenarios = port_scenarios()
    assert len(scenarios) == 6
    assert all(scenario.campaign_stage is None for scenario in scenarios)
    assert scenarios[1].scenario_id == "port-demo-1"
    assert campaign_status()["full_campaign_playable_via_rl"] is False
    assert campaign_status()["campaign"]["rl_partial_stages"] == 2
    assert campaign_status()["campaign"]["rl_proxy_stages"] == 14
    with pytest.raises(ValueError, match="conveyor/bowling"):
        PythonPortConfig(level=4)
    assert PythonPortConfig(campaign_stage="1-1").scenario_id == "1-1"
    assert PythonPortConfig(campaign_stage="1-2").scenario_id == "1-2"
    assert len(supported_campaign_stage_ids()) == 16
    with pytest.raises(ValueError, match="not implemented"):
        PythonPortConfig(campaign_stage="1-5")


def test_proxy_campaign_is_revision_pinned_and_explicitly_reduced() -> None:
    stage = build_campaign_proxy("1-3", seed=7)
    assert stage["fidelity"] == "mechanics-reduced-proxy"
    assert stage["wiki_revision_id"] == 2518396
    assert stage["known_constraints"]["adventure1_wave_count"] == 8
    assert stage["known_constraints"]["available_plants"] == ["Peashooter", "Sunflower"]
    assert stage["known_constraints"]["reward_unlock"] == ["Wall-nut"]
    assert {zombie["name"] for zombie in stage["engine"]["zombie_list"]} <= {
        "Zombie",
        "FlagZombie",
        "ConeheadZombie",
        "BucketheadZombie",
        "NewspaperZombie",
    }
    assert stage["engine"]["background_type"] == 0
    assert build_campaign_proxy("1-4", seed=7)["known_constraints"][
        "available_plants"
    ] == ["Peashooter", "Sunflower", "Wall-nut"]
    assert build_campaign_proxy("2-1", seed=7)["engine"]["background_type"] == 1
    with pytest.raises(ValueError, match="no mechanically reduced proxy"):
        build_campaign_proxy("1-5", seed=7)
