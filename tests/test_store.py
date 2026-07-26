import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pvzwm.data.dataset import SequenceDataset, TransitionDataset
from pvzwm.data.store import ReplayStore, Transition
from pvzwm.training import _to_model_device


def test_replay_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "replay.duckdb"
    expected = Transition(
        np.arange(6, dtype=np.float32),
        3,
        1.25,
        np.arange(6, dtype=np.float32) + 1,
        False,
        legal_mask=np.asarray([True, False, True]),
    )
    with ReplayStore(path) as store:
        with store.episode("test", 7) as episode:
            episode.append(expected)
        assert store.count_transitions() == 1
        actual = next(store.transitions())
    np.testing.assert_array_equal(actual.observation, expected.observation)
    np.testing.assert_array_equal(actual.next_observation, expected.next_observation)
    np.testing.assert_array_equal(actual.legal_mask, expected.legal_mask)
    assert actual.action == expected.action
    assert actual.reward == expected.reward


def test_episode_writer_persists_final_metadata_updates(tmp_path: Path) -> None:
    path = tmp_path / "metadata.duckdb"
    with ReplayStore(path) as store:
        with store.episode("planner", 4, {"won": False}) as episode:
            episode.metadata["won"] = True
        metadata = store.connection.execute(
            "SELECT metadata FROM episodes WHERE episode_id = ?",
            [episode.episode_id],
        ).fetchone()[0]
    decoded = json.loads(metadata) if isinstance(metadata, str) else metadata
    assert decoded["won"] is True


def test_successful_episode_ids_prefer_explicit_outcome_then_return(tmp_path: Path) -> None:
    path = tmp_path / "successes.duckdb"
    with ReplayStore(path) as store:
        cases = [
            (-1.0, {}, False),
            (1.0, {}, True),
            (1.0, {"won": False}, False),
            (-1.0, {"won": True}, True),
        ]
        expected: list[str] = []
        for seed, (reward, metadata, is_success) in enumerate(cases):
            with store.episode("planner", seed, metadata) as episode:
                value = np.asarray([seed], dtype=np.float32)
                episode.append(Transition(value, 0, reward, value, True))
            if is_success:
                expected.append(episode.episode_id)
        assert store.successful_episode_ids("planner") == expected


def test_top_episode_ids_rank_completed_returns(tmp_path: Path) -> None:
    database = tmp_path / "elite.duckdb"
    returns = [-100.0, -20.0, 5.0, -50.0]
    ids: list[str] = []
    with ReplayStore(database) as store:
        for seed, reward in enumerate(returns):
            with store.episode("policy", seed) as writer:
                writer.append(
                    Transition(
                        np.zeros(2, dtype=np.float32),
                        0,
                        reward,
                        np.zeros(2, dtype=np.float32),
                        True,
                        legal_mask=np.ones(1, dtype=np.bool_),
                    )
                )
            ids.append(writer.episode_id)
        assert store.top_episode_ids(0.5, "policy") == [ids[2], ids[1]]
        with pytest.raises(ValueError, match="fraction"):
            store.top_episode_ids(0.0)


def test_failed_episode_marks_its_last_stored_transition_truncated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failed.duckdb"
    with ReplayStore(path) as store:
        with pytest.raises(RuntimeError, match="bridge failed"):
            with store.episode("portable", 7) as episode:
                value = np.zeros(2, dtype=np.float32)
                episode.append(Transition(value, 0, 0.0, value, False))
                raise RuntimeError("bridge failed")
        transition = next(store.transitions())
    assert transition.truncated


def test_new_replay_stores_adjacent_frames_once_without_duplicate_index(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact.duckdb"
    with ReplayStore(path) as store:
        with store.episode("pixels", 3) as episode:
            for step in range(3):
                observation = np.full((3, 4, 5), step, dtype=np.uint8)
                next_observation = np.full((3, 4, 5), step + 1, dtype=np.uint8)
                episode.append(
                    Transition(
                        observation,
                        step,
                        0.0,
                        next_observation,
                        step == 2,
                    )
                )
        stored_next = store.connection.execute(
            "SELECT step, next_observation IS NULL FROM transitions ORDER BY step"
        ).fetchall()
        indexes = store.connection.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'transitions'"
        ).fetchall()
        restored = list(store.transitions())
    assert stored_next == [(0, True), (1, True), (2, False)]
    assert all(name != "transitions_episode_idx" for (name,) in indexes)
    for step, transition in enumerate(restored):
        assert np.all(transition.observation == step)
        assert np.all(transition.next_observation == step + 1)


def test_replay_infers_action_dim_from_legal_mask(tmp_path: Path) -> None:
    path = tmp_path / "actions.duckdb"
    legal = np.zeros(650, dtype=np.bool_)
    legal[[0, 649]] = True
    with ReplayStore(path) as store:
        with store.episode("portable", 1) as episode:
            episode.append(
                Transition(
                    np.zeros(2, dtype=np.float32),
                    649,
                    0.0,
                    np.ones(2, dtype=np.float32),
                    False,
                    legal_mask=legal,
                )
            )
        assert store.infer_action_dim([episode.episode_id], fallback=181) == 650


def test_replay_exposes_next_step_legal_mask_without_crossing_episodes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "next_masks.duckdb"
    masks = [
        np.asarray([True, False, False]),
        np.asarray([True, True, False]),
    ]
    with ReplayStore(path) as store:
        with store.episode("portable", 1) as episode:
            first_episode_id = episode.episode_id
            for step, mask in enumerate(masks):
                value = np.asarray([step], dtype=np.float32)
                episode.append(
                    Transition(value, 0, 0.0, value + 1, step == 1, legal_mask=mask)
                )
        with store.episode("portable", 2) as episode:
            value = np.asarray([9], dtype=np.float32)
            episode.append(
                Transition(value, 0, 0.0, value, True, legal_mask=masks[0])
            )
        transitions = list(store.transitions(episode_ids=[first_episode_id]))
    np.testing.assert_array_equal(transitions[0].next_legal_mask, masks[1])
    assert transitions[1].next_legal_mask is None


def test_replay_sequence_windows_do_not_cross_episodes(tmp_path: Path) -> None:
    path = tmp_path / "sequences.duckdb"
    with ReplayStore(path) as store:
        for seed in range(2):
            with store.episode("test", seed) as episode:
                for step in range(5):
                    value = np.asarray([seed, step], dtype=np.float32)
                    episode.append(Transition(value, 0, 0.0, value + 1, step == 4))
        windows = list(store.transition_sequences(length=3, stride=2))
    assert len(windows) == 4
    assert all(window[0].observation[0] == window[-1].observation[0] for window in windows)


def test_sequence_dataset_includes_legal_masks(tmp_path: Path) -> None:
    path = tmp_path / "legal_sequences.duckdb"
    with ReplayStore(path) as store:
        with store.episode("portable", 0) as episode:
            for step in range(4):
                value = np.asarray([step], dtype=np.float32)
                legal = np.asarray([True, step % 2 == 0, False])
                episode.append(
                    Transition(
                        value,
                        0,
                        0.0,
                        value + 1,
                        step == 3,
                        legal_mask=legal,
                    )
                )
    dataset = SequenceDataset(path, length=3, include_legal_mask=True)
    observations, actions, rewards, next_observations, dones, legal = dataset[0]
    assert observations.shape == next_observations.shape == (3, 1)
    assert actions.shape == rewards.shape == dones.shape == (3,)
    assert legal.shape == (3, 3)
    assert legal.dtype == torch.bool


def test_pixel_dataset_keeps_compact_uint8_storage(tmp_path: Path) -> None:
    path = tmp_path / "pixel_sequences.duckdb"
    with ReplayStore(path) as store:
        with store.episode("pixels", 0) as episode:
            for step in range(3):
                value = np.full((3, 4, 5), step, dtype=np.uint8)
                episode.append(
                    Transition(value, 0, 0.0, value + 1, step == 2)
                )
    transitions = TransitionDataset(path)
    sequences = SequenceDataset(path, length=2)
    assert transitions[0][0].dtype == torch.uint8
    assert transitions[0][3].dtype == torch.uint8
    assert sequences[0][0].dtype == torch.uint8
    assert sequences[0][3].dtype == torch.uint8
    normalized = _to_model_device(
        torch.tensor([0, 128, 255], dtype=torch.uint8), torch.device("cpu")
    )
    assert normalized.dtype == torch.float32
    assert normalized.tolist() == pytest.approx([0.0, 128.0 / 255.0, 1.0])


def test_datasets_treat_time_limit_as_boundary_not_terminal(tmp_path: Path) -> None:
    path = tmp_path / "time_limit.duckdb"
    legal = np.ones(2, dtype=np.bool_)
    with ReplayStore(path) as store:
        with store.episode("portable", 0) as episode:
            for step in range(3):
                value = np.asarray([step], dtype=np.float32)
                episode.append(
                    Transition(
                        value,
                        0,
                        0.0,
                        value + 1,
                        False,
                        truncated=step == 2,
                        legal_mask=legal,
                    )
                )
        with store.episode("portable", 1) as episode:
            value = np.asarray([9], dtype=np.float32)
            episode.append(
                Transition(value, 0, 0.0, value, True, legal_mask=legal)
            )

    transitions = TransitionDataset(path)
    terminal_by_observation = {
        int(observation.item()): float(done.item())
        for observation, _, _, _, done in transitions
    }
    assert terminal_by_observation[2] == 0.0
    assert terminal_by_observation[9] == 1.0

    sequences = SequenceDataset(path, length=3)
    assert len(sequences) == 1
    assert sequences[0][4].tolist() == [0.0, 0.0, 0.0]


def test_episode_split_and_filter_are_episode_safe(tmp_path: Path) -> None:
    path = tmp_path / "split.duckdb"
    with ReplayStore(path) as store:
        for seed in range(5):
            with store.episode("python_port", seed) as episode:
                for step in range(3):
                    value = np.asarray([seed, step], dtype=np.float32)
                    episode.append(Transition(value, 0, 0.0, value + 1, step == 2))
        with store.episode("symbolic", 99) as episode:
            value = np.asarray([99, 0], dtype=np.float32)
            episode.append(Transition(value, 0, 0.0, value, True))

        training, validation = store.split_episode_ids("python_port", 0.4, seed=3)
        assert len(training) == 3
        assert len(validation) == 2
        assert set(training).isdisjoint(validation)
        assert set(training + validation) == set(store.episode_ids("python_port"))
        training_transitions = list(store.transitions(episode_ids=training))
        validation_windows = list(
            store.transition_sequences(2, episode_ids=validation)
        )
    assert len(training_transitions) == 9
    assert len(validation_windows) == 4
    assert {int(item.observation[0]) for item in training_transitions}.isdisjoint(
        {int(window[0].observation[0]) for window in validation_windows}
    )


def test_grouped_episode_split_holds_out_complete_metadata_groups(tmp_path: Path) -> None:
    path = tmp_path / "grouped.duckdb"
    with ReplayStore(path) as store:
        groups_by_episode: dict[str, str] = {}
        for level in ("1-1", "1-2", "1-3"):
            for seed in range(2):
                with store.episode(
                    "pvz_portable_vector", seed, {"level_id": level}
                ) as episode:
                    groups_by_episode[episode.episode_id] = level
                    value = np.asarray([seed], dtype=np.float32)
                    episode.append(Transition(value, 0, 0.0, value, True))
        training, validation = store.split_episode_ids_grouped(
            "level_id", "pvz_portable_vector", 1 / 3, seed=4
        )
    assert len(training) == 4
    assert len(validation) == 2
    assert {groups_by_episode[item] for item in training}.isdisjoint(
        {groups_by_episode[item] for item in validation}
    )


def test_large_pixel_arrays_are_compressed_and_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "pixels.duckdb"
    frame = np.zeros((3, 120, 160), dtype=np.uint8)
    frame[:, 20:100, 20:140] = np.asarray([32, 180, 48], dtype=np.uint8)[:, None, None]
    with ReplayStore(path) as store:
        with store.episode("python_port", 0) as episode:
            episode.append(Transition(frame, 0, 0.0, frame, False))
        stored_bytes = store.connection.execute(
            "SELECT octet_length(observation) + octet_length(next_observation) FROM transitions"
        ).fetchone()[0]
        restored = next(store.transitions()).observation
    assert stored_bytes < frame.nbytes
    np.testing.assert_array_equal(restored, frame)


def test_replay_shards_merge_as_raw_rows_and_reject_duplicates(tmp_path: Path) -> None:
    shards = [tmp_path / "shard_a.duckdb", tmp_path / "shard_b.duckdb"]
    for seed, shard in enumerate(shards):
        with ReplayStore(shard) as store:
            with store.episode("python_port", seed) as episode:
                value = np.full((3, 8, 8), seed, dtype=np.uint8)
                episode.append(Transition(value, seed, float(seed), value + 1, True))

    destination = tmp_path / "merged.duckdb"
    with ReplayStore(destination) as store:
        result = store.merge_from(shards)
        assert result == {
            "databases": 2,
            "episodes": 2,
            "transitions": 2,
            "experiments": 0,
        }
        assert store.count_transitions() == 2
        assert len(store.episode_ids("python_port")) == 2
        with np.testing.assert_raises(ValueError):
            store.merge_from([shards[0]])
