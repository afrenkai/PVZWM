import numpy as np
import pytest

from pvzwm.portable import (
    PortableActionCodec,
    PortableObservation,
    collect_sun_first_mask,
)
from pvzwm.portable_collection import (
    PortableExplorationPolicy,
    encode_portable_observation,
    parse_adventure_levels,
)


def test_parse_adventure_levels_supports_sequential_stage_ranges() -> None:
    assert parse_adventure_levels("1-1..1-3,2-1,50") == [1, 2, 3, 11, 50]
    with pytest.raises(ValueError):
        parse_adventure_levels("5-10..1-1")


def test_exploration_policy_collects_sun_when_available() -> None:
    codec = PortableActionCodec()
    legal = np.zeros(codec.size, dtype=np.bool_)
    legal[[0, codec.plant(0, 0, 0), codec.collect_action]] = True
    assert (
        PortableExplorationPolicy().act(legal, np.random.default_rng(7))
        == codec.collect_action
    )


def test_collect_sun_first_mask_removes_policy_choice_only_while_sun_exists() -> None:
    codec = PortableActionCodec()
    legal = np.zeros(codec.size, dtype=np.bool_)
    plant = codec.plant(0, 0, 0)
    legal[[0, plant, codec.collect_action]] = True
    prioritized = collect_sun_first_mask(codec, legal)
    np.testing.assert_array_equal(np.flatnonzero(prioritized), [codec.collect_action])
    assert legal[[0, plant, codec.collect_action]].all()

    legal[codec.collect_action] = False
    np.testing.assert_array_equal(collect_sun_first_mask(codec, legal), legal)


def test_exploration_policy_only_returns_legal_actions() -> None:
    codec = PortableActionCodec()
    legal = np.zeros(codec.size, dtype=np.bool_)
    legal[[0, codec.plant(0, 0, 0), codec.board_click(1, 2)]] = True
    policy = PortableExplorationPolicy()
    for seed in range(10):
        assert legal[policy.act(legal, np.random.default_rng(seed))]


def test_portable_collection_encodes_explicit_vector_or_pixel_domain() -> None:
    observation = PortableObservation({"area": 1, "stage": 1})
    vector = encode_portable_observation(observation, "vector")
    pixels = np.zeros((3, 12, 16), dtype=np.uint8)
    assert vector.shape == (585,)
    assert encode_portable_observation(observation, "pixels", pixels) is pixels
    with pytest.raises(ValueError, match="rendered frame"):
        encode_portable_observation(observation, "pixels")
