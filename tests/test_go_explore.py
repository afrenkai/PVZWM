import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pvzwm.agents.go_explore import (
    RandomProjectionCellEncoder,
    TrajectoryArchive,
)
from pvzwm.portable import PortableActionCodec

_SPEC = importlib.util.spec_from_file_location(
    "run_portable_go_explore", Path("scripts/run_portable_go_explore.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)


def test_random_projection_cells_are_deterministic() -> None:
    first = RandomProjectionCellEncoder(4, projections=13, seed=7)
    second = RandomProjectionCellEncoder(4, projections=13, seed=7)
    state = np.asarray([1.0, 2.0, -1.0, 0.5], dtype=np.float32)

    assert first.encode(state) == second.encode(state)
    assert len(first.encode(state)) == 26
    with pytest.raises(ValueError):
        first.encode(np.zeros(3, dtype=np.float32))


def test_archive_keeps_higher_return_then_shorter_prefix() -> None:
    archive = TrajectoryArchive()
    key = b"cell"
    first = archive.add(key, 1, [2, 3, 4], -10.0)
    worse = archive.add(key, 2, [5], -20.0)
    better = archive.add(key, 3, [6, 7, 8, 9], -5.0)
    shorter = archive.add(key, 4, [1, 2], -5.0)

    assert first.inserted
    assert not worse.replaced
    assert better.replaced
    assert shorter.replaced
    assert archive.cells[key].seed == 4
    assert archive.cells[key].actions == (1, 2)


def test_archive_round_trip_and_selection_counts() -> None:
    encoder = RandomProjectionCellEncoder(3, projections=8, seed=11)
    archive = TrajectoryArchive()
    first_key = encoder.encode(np.asarray([1.0, 0.0, 0.0]))
    second_key = encoder.encode(np.asarray([0.0, 1.0, 0.0]))
    archive.add(first_key, 10, [], 0.0)
    archive.add(second_key, 10, [4], 1.0, parent_key=first_key)
    selected = archive.select(np.random.default_rng(5))

    restored = TrajectoryArchive.from_payload(archive.payload(encoder), encoder)

    assert selected.selections == 1
    assert restored.cells[first_key].discoveries == 1
    assert restored.cells[selected.key].selections == 1


def test_frontier_selection_prioritizes_deep_productive_cells() -> None:
    archive = TrajectoryArchive()
    archive.add(b"a", 1, [1], 0.0)
    archive.add(b"b", 1, list(range(20)), 0.0)
    productive = archive.add(b"c", 1, list(range(21)), 0.0).cell
    productive.discoveries = 8

    weights = archive.selection_weights("frontier", frontier_fraction=0.5)

    assert weights.shape == (3,)
    assert weights[0] < weights[1] < weights[2]
    assert np.isclose(weights.sum(), 1.0)


def test_frontier_selection_validates_configuration() -> None:
    archive = TrajectoryArchive()
    archive.add(b"a", 1, [], 0.0)

    with pytest.raises(ValueError, match="strategy"):
        archive.selection_weights("unknown")
    with pytest.raises(ValueError, match="frontier_fraction"):
        archive.selection_weights("frontier", 0.0)


def test_frontier_branch_selection_boosts_rare_action_branches() -> None:
    archive = TrajectoryArchive()
    for index in range(9):
        archive.add(bytes([index]), 1, list(range(20)), 0.0)
    archive.add(b"rare", 1, list(range(20)), 0.0)

    weights = archive.selection_weights(
        "frontier-branch",
        frontier_fraction=1.0,
        branch_ids=["common"] * 9 + ["rare"],
    )

    assert weights[-1] > weights[0]
    assert np.isclose(weights.sum(), 1.0)


def test_frontier_branch_selection_requires_aligned_branch_ids() -> None:
    archive = TrajectoryArchive()
    archive.add(b"a", 1, [], 0.0)

    with pytest.raises(ValueError, match="branch id"):
        archive.selection_weights("frontier-branch", branch_ids=[])


def test_factored_random_groups_remove_coordinate_cardinality_bias() -> None:
    codec = PortableActionCodec()
    legal = np.zeros(codec.size, dtype=np.bool_)
    legal[0] = True
    legal[codec.plant(0, 0, 0)] = True
    legal[codec.plant(0, 0, 1)] = True
    legal[codec.plant(2, 4, 3)] = True

    groups = _SCRIPT.factored_random_action_groups(legal, codec)

    assert [group.tolist() for group in groups] == [
        [0],
        [codec.plant(0, 0, 0), codec.plant(0, 0, 1)],
        [codec.plant(2, 4, 3)],
    ]


def test_action_branch_groups_plant_coordinates_by_seed_slot() -> None:
    codec = PortableActionCodec()

    assert _SCRIPT.action_branch(codec.plant(2, 0, 0), codec) == ("plant", 2)
    assert _SCRIPT.action_branch(codec.plant(2, 5, 8), codec) == ("plant", 2)
    assert _SCRIPT.action_branch(0, codec) == ("wait", 0)


def test_archive_branch_trace_retains_recent_rare_action() -> None:
    codec = PortableActionCodec()
    archive = TrajectoryArchive()
    actions = [0, codec.plant(2, 2, 3), 0]
    archive.add(b"trace", 1, actions, 0.0)

    latest_only = _SCRIPT.archive_branch_ids(archive, codec, trace_window=1)
    recent_trace = _SCRIPT.archive_branch_ids(archive, codec, trace_window=3)

    assert latest_only == [(('wait', 0), (('wait', 0),))]
    assert recent_trace == [
        (('wait', 0), (('plant', 2), ('wait', 0)))
    ]


def test_archive_branch_trace_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="trace window"):
        _SCRIPT.archive_branch_ids(
            TrajectoryArchive(), PortableActionCodec(), trace_window=0
        )


def test_factored_random_action_is_always_legal() -> None:
    codec = PortableActionCodec()
    legal = np.zeros(codec.size, dtype=np.bool_)
    legal[[0, codec.plant(1, 2, 3), codec.shovel(4, 5)]] = True
    generator = np.random.default_rng(19)

    actions = {
        _SCRIPT.random_legal_action(legal, codec, generator, "factored")
        for _ in range(100)
    }

    assert actions == set(np.flatnonzero(legal))


def test_restore_can_replace_unavailable_collect_with_wait() -> None:
    codec = PortableActionCodec()
    legal = np.zeros(codec.size, dtype=np.bool_)
    legal[0] = True

    action, substituted = _SCRIPT.resolve_restore_action(
        codec.collect_action, legal, codec, True
    )

    assert action == 0
    assert substituted


def test_restore_can_replace_wait_with_newly_available_collect() -> None:
    codec = PortableActionCodec()
    legal_mask = np.zeros(codec.size, dtype=np.bool_)
    legal_mask[0] = True
    legal_mask[codec.collect_action] = True
    action, substituted = _SCRIPT.resolve_restore_action(
        0, legal_mask, codec, tolerate_collect_timing=True
    )
    assert action == codec.collect_action
    assert substituted


def test_restore_does_not_repair_non_housekeeping_actions() -> None:
    codec = PortableActionCodec()
    legal = np.zeros(codec.size, dtype=np.bool_)
    legal[0] = True

    action, substituted = _SCRIPT.resolve_restore_action(
        codec.plant(2, 2, 0), legal, codec, True
    )

    assert action is None
    assert not substituted
