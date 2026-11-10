# solver/model.py
from __future__ import annotations

import torch
import torch.nn as nn


class CubeNet(nn.Module):
    """큐브 상태 -> (거리 분포, 18개 이동 확률) 동시 예측.
    입력: one-hot (N, 324)  = 54 facelet x 6색
    출력:
        dist_logits  : (N, n_classes)  - 완성까지 남은 회전 수 분포
        policy_logits: (N, 18)         - 18개 이동의 선호도
    """

    def __init__(self, n_classes: int = 21, hidden: int = 512,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        d = 324
        for _ in range(n_layers):
            layers += [
                nn.Linear(d, hidden),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            d = hidden
        self.encoder = nn.Sequential(*layers)

        self.dist_head = nn.Linear(d, n_classes)
        self.n_classes = n_classes
        self.register_buffer("depth_values",
                             torch.arange(n_classes, dtype=torch.float32))

        self.policy_head = nn.Linear(d, 18)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.dist_head(h), self.policy_head(h)

    @torch.no_grad()
    def expected_distance(self, x: torch.Tensor) -> torch.Tensor:
        """거리 예측 기댓값 (N,). 탐색 휴리스틱용."""
        dist_logits, _ = self.forward(x)
        p = torch.softmax(dist_logits, dim=-1)
        return (p * self.depth_values).sum(dim=-1)

    @torch.no_grad()
    def policy_prob(self, x: torch.Tensor) -> torch.Tensor:
        """18개 이동 확률 (N, 18)."""
        _, policy_logits = self.forward(x)
        return torch.softmax(policy_logits, dim=-1)
