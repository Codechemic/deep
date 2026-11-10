# solver/solve.py
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from cube.cube import MOVES, MOVE_PERMS, is_solved, one_hot

_FACE_OF_MOVE = np.array([i // 3 for i in range(len(MOVES))])


class Solver:
    def __init__(self, weights_dir: str, device: str = "cpu"):
        from solver.model import CubeNet

        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        ckpt = torch.load(Path(weights_dir) / "model.pt", map_location=self.device)
        self.model = CubeNet(**ckpt["config"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def heuristic(self, states: np.ndarray) -> np.ndarray:
        """(N, 54) -> (N,) predicted remaining moves (smaller = closer to solved)."""
        x = torch.from_numpy(one_hot(states)).to(self.device)
        return self.model.expected_distance(x).cpu().numpy()

    @torch.no_grad()
    def policy_scores(self, states: np.ndarray, move_idx: np.ndarray) -> np.ndarray:
        """각 상태의 선택한 이동에 대한 정책 확률 (N,)."""
        x = torch.from_numpy(one_hot(states)).to(self.device)
        policy = self.model.policy_prob(x)                 # (N, 18)
        return policy[np.arange(len(policy)), move_idx].cpu().numpy()

    def solve(self, state: list[int], time_limit: float,
              max_len: int = 60, beam_width: int = 30,
              policy_weight: float = 1.0) -> list[str]:
        """정책 가중치 기본 1.0 (학습된 정책을 탐색에 적극 반영)."""
        t_end = time.perf_counter() + time_limit * 0.9
        cur = np.asarray(state, dtype=np.int8)
        if is_solved(cur):
            return []

        # beam_states: list of (state, path, last_face, g)
        beam_states = [(cur, [], -1, 0)]
        visited = {cur.tobytes()}

        for _ in range(max_len):
            if time.perf_counter() > t_end:
                break

            # expand all beam states: collect valid children
            cand_states = []          # (M, 54)
            cand_meta = []            # (b_state, b_path, move_idx, b_g)
            for b_state, b_path, b_last_face, b_g in beam_states:
                cand = b_state[MOVE_PERMS]          # (18, 54)
                for i in range(len(MOVES)):
                    new_state = cand[i].copy()
                    new_bytes = new_state.tobytes()
                    if new_bytes in visited:
                        continue
                    if _FACE_OF_MOVE[i] == b_last_face:
                        continue
                    visited.add(new_bytes)
                    cand_states.append(new_state)
                    cand_meta.append((b_state, b_path, i, b_g))

            if not cand_states:
                break

            cand_states = np.array(cand_states)
            h = self.heuristic(cand_states)            # (M,)

            # 정책 점수는 policy_weight > 0 일 때만 반영
            ranked = []
            for k, (b, b_path, i, b_g) in enumerate(cand_meta):
                ns = cand_states[k]
                f = (b_g + 1) + h[k]
                if policy_weight > 0:
                    p = self.policy_scores(np.array([ns]), np.array([i]))[0]
                    f = f - policy_weight * p
                ranked.append((f, ns, b_path + [MOVES[i]], _FACE_OF_MOVE[i], b_g + 1))

            ranked.sort(key=lambda x: x[0])
            beam_states = [(s[1], s[2], s[3], s[4]) for s in ranked[:beam_width]]

            # early exit: solved state in beam
            for st, path, _, _ in beam_states:
                if is_solved(st):
                    return path

        return []
