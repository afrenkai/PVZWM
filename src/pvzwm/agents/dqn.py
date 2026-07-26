from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


@dataclass(slots=True)
class DQNBatch:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    next_legal_mask: torch.Tensor | None = None


class MaskedDoubleDQN:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        discount: float = 0.99,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.online = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.discount = discount

    @torch.inference_mode()
    def act(self, state: torch.Tensor, legal_mask: torch.Tensor, epsilon: float = 0.0) -> int:
        if torch.rand((), device=self.device) < epsilon:
            legal = torch.nonzero(legal_mask.to(self.device), as_tuple=False).flatten()
            return int(
                legal[torch.randint(len(legal), (), device=self.device)].item()
            )
        values = self.online(state.to(self.device).unsqueeze(0)).squeeze(0)
        values = values.masked_fill(~legal_mask.to(self.device), -torch.inf)
        return int(values.argmax().item())

    def loss(self, batch: DQNBatch) -> torch.Tensor:
        state = batch.state.to(self.device)
        action = batch.action.to(self.device)
        reward = batch.reward.to(self.device)
        next_state = batch.next_state.to(self.device)
        done = batch.done.to(self.device)
        current = self.online(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online = self.online(next_state)
            if batch.next_legal_mask is not None:
                next_online = next_online.masked_fill(
                    ~batch.next_legal_mask.to(self.device), -torch.inf
                )
            next_action = next_online.argmax(dim=1, keepdim=True)
            next_value = self.target(next_state).gather(1, next_action).squeeze(1)
            target = reward + self.discount * (1.0 - done) * next_value
        return F.smooth_l1_loss(current, target)

    def update_target(self, tau: float = 1.0) -> None:
        with torch.no_grad():
            for target, online in zip(self.target.parameters(), self.online.parameters(), strict=True):
                target.lerp_(online, tau)
