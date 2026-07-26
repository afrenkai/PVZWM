from __future__ import annotations

from dataclasses import asdict, dataclass

from pvzwm.campaign_proxy import PROXY_CAMPAIGN_STAGE_IDS


@dataclass(frozen=True, slots=True)
class CampaignStage:
    stage_id: str
    world: int
    level: int
    area: str
    rows: int
    mode: str
    implementation: str = "missing"


@dataclass(frozen=True, slots=True)
class PortScenario:
    scenario_id: str
    upstream_level_file: str
    area: str
    mode: str
    upstream_playable: bool
    rl_status: str
    campaign_stage: None = None


_AREA_BY_WORLD = {
    1: "day",
    2: "night",
    3: "pool",
    4: "fog",
    5: "roof",
}


def campaign_stages() -> list[CampaignStage]:
    """Return the ordered PvZ1 Adventure 1-1 through 5-10 curriculum."""
    stages: list[CampaignStage] = []
    for world in range(1, 6):
        for level in range(1, 11):
            area = _AREA_BY_WORLD[world]
            if world == 4 and level == 5:
                area = "night"
            elif world == 5 and level == 10:
                area = "night-roof"
            mode = "standard"
            if world == 1 and level <= 3:
                mode = "tutorial"
            if level == 5:
                mode = "bonus"
            if level == 10:
                mode = "boss" if world == 5 else "conveyor"
            stages.append(
                CampaignStage(
                    stage_id=f"{world}-{level}",
                    world=world,
                    level=level,
                    area=area,
                    rows=6 if area in {"pool", "fog"} else 5,
                    mode=mode,
                    implementation=(
                        "partial"
                        if world == 1 and level in {1, 2}
                        else "proxy"
                        if f"{world}-{level}" in PROXY_CAMPAIGN_STAGE_IDS
                        else "missing"
                    ),
                )
            )
    return stages


def supported_campaign_stage_ids() -> tuple[str, ...]:
    return ("1-1", "1-2", *PROXY_CAMPAIGN_STAGE_IDS)


def port_scenarios() -> list[PortScenario]:
    """Describe upstream demo files without equating them to campaign stages."""
    return [
        PortScenario("port-demo-0", "level_0.json", "day", "static", True, "supported"),
        PortScenario("port-demo-1", "level_1.json", "day", "static", True, "supported"),
        PortScenario("port-demo-2", "level_2.json", "day", "static", True, "supported"),
        PortScenario(
            "port-demo-3", "level_3.json", "night", "static", True, "partial-mechanics"
        ),
        PortScenario(
            "port-demo-4", "level_4.json", "day", "conveyor", True, "adapter-missing"
        ),
        PortScenario(
            "port-demo-5", "level_5.json", "day", "wallnut-bowling", True, "adapter-missing"
        ),
    ]


def campaign_status() -> dict[str, object]:
    stages = campaign_stages()
    scenarios = port_scenarios()
    return {
        "campaign": {
            "format": "1-1 through 5-10",
            "total_stages": len(stages),
            "rl_playable_stages": sum(stage.implementation == "playable" for stage in stages),
            "rl_partial_stages": sum(stage.implementation == "partial" for stage in stages),
            "rl_proxy_stages": sum(stage.implementation == "proxy" for stage in stages),
            "world_model_coverage": "port-demos-plus-partial-and-mechanics-reduced-proxies",
            "stages": [asdict(stage) for stage in stages],
        },
        "python_port": {
            "scenario_count": len(scenarios),
            "rl_supported_scenarios": sum(
                scenario.rl_status in {"supported", "partial-mechanics"}
                for scenario in scenarios
            ),
            "scenarios": [asdict(scenario) for scenario in scenarios],
        },
        "full_campaign_playable_via_rl": False,
    }
