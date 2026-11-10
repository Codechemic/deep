"""스크램블(섞기) 생성기와 문제 세트 생성.

문제(problem) 형식
------------------
{
  "problem_id": "d05-0003",       # 구간-일련번호
  "tier": "d01-05",               # 난이도 구간 이름
  "scramble_depth": 5,            # 섞은 회전 수 (random 구간은 null)
  "state": [0, 0, 2, ...]         # 길이 54 정수 배열
}

학생 배포용 공개 세트에는 scramble(섞은 회전 목록)도 함께 넣어 참고하게 하고,
채점용 비공개 세트에는 state만 넣는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cube import FACES, MOVES, apply_moves, solved_state

# 난이도 구간 정의: (이름, 최소 깊이, 최대 깊이). random 은 사실상 완전 무작위 상태.
DEFAULT_TIERS: list[tuple[str, int, int]] = [
    ("d01-05", 1, 5),
    ("d06-10", 6, 10),
    ("d11-15", 11, 15),
    ("d16-20", 16, 20),
    ("random", 60, 60),
]

_OPPOSITE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}


def random_scramble(depth: int, rng: np.random.Generator) -> list[str]:
    """같은 면 연속 회전과 '반대 면 사이에 낀 같은 면' 패턴을 피하는 무작위 스크램블."""
    moves: list[str] = []
    while len(moves) < depth:
        m = MOVES[int(rng.integers(len(MOVES)))]
        face = m[0]
        if moves and moves[-1][0] == face:
            continue
        if (len(moves) >= 2 and moves[-2][0] == face
                and moves[-1][0] == _OPPOSITE[face]):
            continue
        moves.append(m)
    return moves


def make_problem_set(
    n_per_tier: int,
    seed: int,
    tiers: list[tuple[str, int, int]] | None = None,
    include_scramble: bool = False,
) -> list[dict]:
    tiers = tiers or DEFAULT_TIERS
    rng = np.random.default_rng(seed)
    problems = []
    for name, lo, hi in tiers:
        for i in range(n_per_tier):
            depth = int(rng.integers(lo, hi + 1))
            scr = random_scramble(depth, rng)
            state = apply_moves(solved_state(), scr)
            p = {
                "problem_id": f"{name}-{i:04d}",
                "tier": name,
                "scramble_depth": None if name == "random" else depth,
                "state": [int(x) for x in state],
            }
            if include_scramble:
                p["scramble"] = scr
            problems.append(p)
    return problems


def save_problem_set(problems: list[dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps(problems, ensure_ascii=False, indent=1), encoding="utf-8")


def load_problem_set(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="문제 세트 생성")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-tier", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--include-scramble", action="store_true", help="섞은 회전 목록 포함(공개용)")
    a = ap.parse_args()
    ps = make_problem_set(a.n_per_tier, a.seed, include_scramble=a.include_scramble)
    save_problem_set(ps, a.out)
    print(f"{len(ps)}개 문제 저장: {a.out}")
