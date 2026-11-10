"""3x3x3 큐브 시뮬레이터 (facelet 표현).

상태 표현
---------
길이 54의 정수 배열. 각 원소는 그 위치(facelet)에 붙어 있는 스티커의 색 (0~5).
색 번호는 "완성 상태에서 그 스티커가 속한 면"과 같다.

    면 순서 : U=0, R=1, F=2, D=3, L=4, B=5
    인덱스  : U: 0~8, R: 9~17, F: 18~26, D: 27~35, L: 36~44, B: 45~53

각 면 안에서는 바깥에서 그 면을 바라봤을 때 왼쪽 위부터 오른쪽 아래까지 행 우선 순서.
(Kociemba facelet 표기와 동일한 순서이므로 외부 도구와 호환된다.)

                 |------------|
                 |  0   1   2 |
                 |  3   4   5 |      U
                 |  6   7   8 |
    |------------|------------|------------|------------|
    | 36  37  38 | 18  19  20 |  9  10  11 | 45  46  47 |
    | 39  40  41 | 21  22  23 | 12  13  14 | 48  49  50 |
    | 42  43  44 | 24  25  26 | 15  16  17 | 51  52  53 |
    |     L      |     F      |     R      |     B      |
                 |------------|
                 | 27  28  29 |
                 | 30  31  32 |      D
                 | 33  34  35 |
                 |------------|

회전 표기
---------
Singmaster 표기. 면 문자 U, R, F, D, L, B 뒤에
  (없음) : 시계 방향 90도   (그 면을 바깥에서 바라봤을 때)
  '      : 반시계 방향 90도
  2      : 180도
총 18가지 회전이 허용된다 (half-turn metric).
"""
from __future__ import annotations

import numpy as np

FACES = "URFDLB"
FACE_INDEX = {f: i for i, f in enumerate(FACES)}
MOVES: list[str] = [f + s for f in FACES for s in ("", "'", "2")]
MOVE_INDEX = {m: i for i, m in enumerate(MOVES)}
N_FACELETS = 54
SOLVED_STATE = np.repeat(np.arange(6, dtype=np.int8), 9)


def _idx(face: str, n: int) -> int:
    """면 문자와 1~9 번호로 facelet 인덱스 계산 (U1 -> 0, R1 -> 9 ...)."""
    return FACE_INDEX[face] * 9 + (n - 1)


# 각 기본(시계 방향 90도) 회전을 4-cycle 목록으로 정의한다.
# (a, b, c, d) 는 "a 위치의 스티커가 b 위치로, b는 c로, c는 d로, d는 a로 이동"을 뜻한다.
_CYCLES: dict[str, list[tuple[str, ...]]] = {
    "U": [("U1", "U3", "U9", "U7"), ("U2", "U6", "U8", "U4"),
          ("F1", "L1", "B1", "R1"), ("F2", "L2", "B2", "R2"), ("F3", "L3", "B3", "R3")],
    "D": [("D1", "D3", "D9", "D7"), ("D2", "D6", "D8", "D4"),
          ("F7", "R7", "B7", "L7"), ("F8", "R8", "B8", "L8"), ("F9", "R9", "B9", "L9")],
    "F": [("F1", "F3", "F9", "F7"), ("F2", "F6", "F8", "F4"),
          ("U7", "R1", "D3", "L9"), ("U8", "R4", "D2", "L6"), ("U9", "R7", "D1", "L3")],
    "B": [("B1", "B3", "B9", "B7"), ("B2", "B6", "B8", "B4"),
          ("U3", "L1", "D7", "R9"), ("U2", "L4", "D8", "R6"), ("U1", "L7", "D9", "R3")],
    "R": [("R1", "R3", "R9", "R7"), ("R2", "R6", "R8", "R4"),
          ("U9", "B1", "D9", "F9"), ("U6", "B4", "D6", "F6"), ("U3", "B7", "D3", "F3")],
    "L": [("L1", "L3", "L9", "L7"), ("L2", "L6", "L8", "L4"),
          ("U1", "F1", "D1", "B9"), ("U4", "F4", "D4", "B6"), ("U7", "F7", "D7", "B3")],
}


def _perm_from_cycles(cycles: list[tuple[str, ...]]) -> np.ndarray:
    """new_state = state[perm] 꼴로 쓸 수 있는 순열 배열 생성."""
    perm = np.arange(N_FACELETS)
    for cyc in cycles:
        ids = [_idx(tok[0], int(tok[1])) for tok in cyc]
        for src, dst in zip(ids, ids[1:] + ids[:1]):
            perm[dst] = src
    return perm


def _compose(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """p 적용 후 q 적용과 같은 순열. state[p][q] == state[p[q]]"""
    return p[q]


def _build_move_perms() -> np.ndarray:
    perms = np.zeros((len(MOVES), N_FACELETS), dtype=np.int64)
    for face, cycles in _CYCLES.items():
        p1 = _perm_from_cycles(cycles)
        p2 = _compose(p1, p1)
        p3 = _compose(p2, p1)
        perms[MOVE_INDEX[face]] = p1
        perms[MOVE_INDEX[face + "2"]] = p2
        perms[MOVE_INDEX[face + "'"]] = p3
    return perms


MOVE_PERMS: np.ndarray = _build_move_perms()  # shape (18, 54)

# 각 회전의 역회전 인덱스
INVERSE_MOVE: dict[str, str] = {}
for _f in FACES:
    INVERSE_MOVE[_f] = _f + "'"
    INVERSE_MOVE[_f + "'"] = _f
    INVERSE_MOVE[_f + "2"] = _f + "2"


# ----------------------------------------------------------------------------
# 공개 함수
# ----------------------------------------------------------------------------
def solved_state() -> np.ndarray:
    return SOLVED_STATE.copy()


def is_solved(state: np.ndarray | list[int]) -> bool:
    return bool(np.array_equal(np.asarray(state, dtype=np.int8), SOLVED_STATE))


def apply_move(state: np.ndarray | list[int], move: str) -> np.ndarray:
    """단일 회전을 적용한 새 상태를 반환한다 (원본은 바꾸지 않음)."""
    if move not in MOVE_INDEX:
        raise ValueError(f"허용되지 않는 회전: {move!r}")
    return np.asarray(state, dtype=np.int8)[MOVE_PERMS[MOVE_INDEX[move]]]


def apply_moves(state: np.ndarray | list[int], moves: list[str] | str) -> np.ndarray:
    """회전 목록(또는 공백으로 구분한 문자열)을 순서대로 적용한다."""
    if isinstance(moves, str):
        moves = parse_moves(moves)
    s = np.asarray(state, dtype=np.int8)
    for m in moves:
        s = apply_move(s, m)
    return s


def apply_moves_batch(states: np.ndarray, move_idx: int) -> np.ndarray:
    """(N, 54) 상태 배치에 같은 회전을 한 번에 적용한다. 탐색 가속용."""
    return states[:, MOVE_PERMS[move_idx]]


def expand_all(state: np.ndarray) -> np.ndarray:
    """한 상태에 18가지 회전을 모두 적용한 (18, 54) 배열."""
    return np.asarray(state, dtype=np.int8)[MOVE_PERMS]


def parse_moves(text: str) -> list[str]:
    moves = text.split()
    for m in moves:
        if m not in MOVE_INDEX:
            raise ValueError(f"허용되지 않는 회전: {m!r}")
    return moves


def inverse_moves(moves: list[str]) -> list[str]:
    return [INVERSE_MOVE[m] for m in reversed(moves)]


def to_facelet_string(state: np.ndarray | list[int]) -> str:
    """Kociemba 형식 문자열 (예: 'UUUUUUUUURRR...') 로 변환."""
    return "".join(FACES[int(c)] for c in state)


def from_facelet_string(text: str) -> np.ndarray:
    if len(text) != N_FACELETS:
        raise ValueError("facelet 문자열 길이는 54여야 합니다")
    return np.array([FACE_INDEX[ch] for ch in text], dtype=np.int8)


def one_hot(states: np.ndarray) -> np.ndarray:
    """(N, 54) 정수 상태 -> (N, 324) float32 one-hot. 신경망 입력용."""
    states = np.asarray(states, dtype=np.int64)
    if states.ndim == 1:
        states = states[None, :]
    n = states.shape[0]
    out = np.zeros((n, N_FACELETS * 6), dtype=np.float32)
    flat_idx = np.arange(N_FACELETS) * 6 + states
    out[np.arange(n)[:, None], flat_idx] = 1.0
    return out


def is_valid_state(state: np.ndarray | list[int]) -> bool:
    """색 개수와 중심 스티커만 검사하는 가벼운 유효성 검사."""
    s = np.asarray(state)
    if s.shape != (N_FACELETS,):
        return False
    if not np.all((s >= 0) & (s <= 5)):
        return False
    if not np.all(np.bincount(s, minlength=6) == 9):
        return False
    centers = s[[4, 13, 22, 31, 40, 49]]
    return bool(np.array_equal(centers, np.arange(6)))


def pretty(state: np.ndarray | list[int]) -> str:
    """전개도 형태의 문자열."""
    s = to_facelet_string(state)
    f = lambda face, r: " ".join(s[FACE_INDEX[face] * 9 + r * 3: FACE_INDEX[face] * 9 + r * 3 + 3])
    lines = []
    for r in range(3):
        lines.append(" " * 8 + f("U", r))
    for r in range(3):
        lines.append("  ".join([f("L", r), f("F", r), f("R", r), f("B", r)]))
    for r in range(3):
        lines.append(" " * 8 + f("D", r))
    return "\n".join(lines)
