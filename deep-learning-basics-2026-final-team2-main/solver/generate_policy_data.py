# solver/generate_policy_data.py
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import kociemba

from cube.cube import MOVES, solved_state, scramble, one_hot

# ----- kociemba 출력과 MOVES 매칭 -----
# MOVES = ['U', "U'", 'U2', 'R', "R'", 'R2', 'F', "F'", 'F2',
#          'D', "D'", 'D2', 'L', "L'", 'L2', 'B', "B'", 'B2']
# kociemba.solve 첫 토큰이 이 MOVES 중 하나와 정확히 일치해야 함.

def _token_matches_move(move: str, token: str) -> bool:
    """kociemba 첫 토큰(예: "R")이 MOVES[move_idx]의 move와 일치하는지."""
    if not token or not move:
        return False
    if move[0] != token[0]:
        return False
    # 방향 문자 비교 (예: "R"→"", "R'"→"'", "R2"→"2")
    token_dir = token[1:] if len(token) > 1 else ""
    move_dir = move[1:] if len(move) > 1 else ""
    return token_dir == move_dir


def kociemba_first_move_index(state: np.ndarray) -> int | None:
    """상태를 kociemba로 풀어 첫 번째 최적 회전(MOVES index) 반환. 못 풀면 None."""
    s = _state_to_kociemba_str(state)
    try:
        sol = kociemba.solve(s)
    except Exception:
        return None
    if not sol or sol.strip() == "":
        return None
    first_token = sol.strip().split()[0]   # 예: "R", "U'", "F2"
    for i, m in enumerate(MOVES):
        if _token_matches_move(m, first_token):
            return i
    return None


def _state_to_kociemba_str(state: np.ndarray) -> str:
    """(54,) int 배열을 kociemba 입력 문자열(6면 x 9개)로 변환.

    ⚠️ color_map / 면 순서가 cube.cube.py 실제 정의와 일치해야 함.
    아래는 표준 WCA 순서의 예시. cube.cube와 다르면 이 함수를 수정해야 함.
    """
    # color_map: cube.cube의 색 인덱스 → 문자 (U/R/F/B/L/D)
    # cube.cube에서 "0=U, 1=R, 2=F, 3=B, 4=L, 5=D" 인지 확인 필요.
    # 아래는 가장 흔한 매핑 가정.
    color_map = {0: "U", 1: "R", 2: "F", 3: "B", 4: "L", 5: "D"}

    # 면 순서: kociemba 표준 입력은 보통 U R F D L B 순서.
    # cube.cube의 facelet 순서(0-8, 9-17, ...)가 이 순서와 일치하는지 확인 필요.
    face_order = [0, 1, 2, 5, 4, 3]  # U, R, F, D, L, B (cube.cube 순서와 맞춰야 함)

    out = []
    for fc in face_order:
        for i in range(9):
            idx = fc * 9 + i
            out.append(color_map[int(state[idx])])
    return "".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="solver/data/policy_dataset.npz",
                    help="저장 경로 (예: solver/data/policy_dataset.npz)")
    ap.add_argument("--n-samples", type=int, default=50000,
                    help="생성할 샘플 수")
    ap.add_argument("--max-depth", type=int, default=25,
                    help="무작위 스크램블 최대 깊이")
    ap.add_argument("--seed", type=int, default=42,
                    help="난수 시드")
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    states: list[np.ndarray] = []
    policy_labels: list[int] = []

    t0 = time.time()
    for i in range(a.n_samples):
        d = int(rng.integers(1, a.max_depth + 1))
        state = scramble(d, seed=int(rng.integers(0, 2**31)))
        mv = kociemba_first_move_index(state)
        if mv is not None:
            states.append(state)
            policy_labels.append(mv)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"{i + 1}/{a.n_samples}  정책라벨 확보 {len(states)}개  {elapsed:.0f}s",
                  flush=True)

    states = np.array(states, dtype=np.int8)
    policy_labels = np.array(policy_labels, dtype=np.int64)
    x = one_hot(states)   # (N, 324)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=x, policy=policy_labels)
    size_mb = out.stat().st_size / 1e6
    print(f"\n저장: {out}  ({size_mb:.1f} MB)  "
          f"상태 {states.shape[0]}개, 정책 라벨 확보")
    print(f"생성 완료: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
