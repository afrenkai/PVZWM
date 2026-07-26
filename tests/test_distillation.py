import numpy as np
import pytest
import torch

from pvzwm.agents.distillation import (
    balanced_level_batch_indices,
    masked_policy_distillation_loss,
)


def test_masked_policy_distillation_matches_teacher_and_ignores_illegal_logits() -> None:
    teacher = torch.tensor([[2.0, -10.0, 0.5]])
    legal = torch.tensor([[True, False, True]])
    matching_student = teacher.clone()
    altered_illegal = teacher.clone()
    altered_illegal[:, 1] = 1000.0
    assert masked_policy_distillation_loss(matching_student, teacher, legal) < 1e-7
    assert masked_policy_distillation_loss(altered_illegal, teacher, legal) < 1e-7
    shifted = torch.tensor([[0.0, -10.0, 2.0]])
    assert masked_policy_distillation_loss(shifted, teacher, legal) > 0.1


def test_balanced_level_sampling_equalizes_unequal_teacher_corpora() -> None:
    selections = balanced_level_batch_indices(
        [3, 30, 300], 32, np.random.default_rng(9)
    )
    assert sorted(len(selection) for selection in selections) == [10, 11, 11]
    assert all((selection >= 0).all() for selection in selections)
    assert all(
        (selection < size).all()
        for selection, size in zip(selections, [3, 30, 300])
    )
    with pytest.raises(ValueError):
        balanced_level_batch_indices([2, 2, 2], 2, np.random.default_rng(1))
