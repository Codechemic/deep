"""시뮬레이터 정합성 검사. `pytest tests/` 로 실행."""
import numpy as np
import pytest

from cube.cube import (FACES, MOVES, apply_move, apply_moves, inverse_moves, is_solved,
                       is_valid_state, one_hot, solved_state, to_facelet_string,
                       from_facelet_string)


def _order(seq: str, limit: int = 2000) -> int:
    s = solved_state()
    for k in range(1, limit + 1):
        s = apply_moves(s, seq)
        if is_solved(s):
            return k
    raise AssertionError(f"{seq} 의 차수가 {limit} 이내에 없음")


def test_each_move_has_order_4_or_2():
    for f in FACES:
        assert _order(f) == 4
        assert _order(f + "'") == 4
        assert _order(f + "2") == 2


def test_inverse_cancels():
    rng = np.random.default_rng(0)
    for _ in range(50):
        seq = [MOVES[i] for i in rng.integers(len(MOVES), size=12)]
        s = apply_moves(solved_state(), seq)
        assert is_solved(apply_moves(s, inverse_moves(seq)))


def test_opposite_faces_commute():
    for a, b in [("U", "D"), ("R", "L"), ("F", "B")]:
        s1 = apply_moves(solved_state(), [a, b])
        s2 = apply_moves(solved_state(), [b, a])
        assert np.array_equal(s1, s2)


def test_known_group_orders():
    # 큐브 군의 잘 알려진 원소 차수. 순열 정의가 하나라도 틀리면 거의 확실히 깨진다.
    assert _order("R U R' U'") == 6
    assert _order("R U") == 105
    assert _order("R U2 D' B D'") == 1260
    assert _order("F R U R' U' F'") == 6


def test_every_move_keeps_state_valid():
    rng = np.random.default_rng(1)
    s = solved_state()
    for i in rng.integers(len(MOVES), size=200):
        s = apply_move(s, MOVES[i])
        assert is_valid_state(s)


def test_facelet_string_roundtrip():
    s = apply_moves(solved_state(), "R U F' D2 L B")
    assert np.array_equal(from_facelet_string(to_facelet_string(s)), s)
    assert to_facelet_string(solved_state()) == "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9


def test_one_hot_shape():
    x = one_hot(np.stack([solved_state(), apply_move(solved_state(), "U")]))
    assert x.shape == (2, 324)
    assert np.all(x.sum(axis=1) == 54)


def test_invalid_move_rejected():
    with pytest.raises(ValueError):
        apply_move(solved_state(), "X")


def test_against_kociemba_solver():
    """외부 솔버가 낸 해법을 우리 시뮬레이터로 적용했을 때 완성되어야 한다."""
    kociemba = pytest.importorskip("kociemba", reason="kociemba 미설치")
    rng = np.random.default_rng(7)
    for _ in range(5):
        seq = [MOVES[i] for i in rng.integers(len(MOVES), size=30)]
        s = apply_moves(solved_state(), seq)
        sol = kociemba.solve(to_facelet_string(s))
        assert is_solved(apply_moves(s, sol))


# 모서리 조각의 두 facelet 쌍 (12개 엣지)
_EDGES = [("U2", "B2"), ("U4", "L2"), ("U6", "R2"), ("U8", "F2"),
          ("D2", "F8"), ("D4", "L8"), ("D6", "R8"), ("D8", "B8"),
          ("F4", "L6"), ("F6", "R4"), ("B4", "R6"), ("B6", "L4")]


def _i(tok):
    return "URFDLB".index(tok[0]) * 9 + int(tok[1]) - 1


def test_superflip_flips_every_edge_only():
    """슈퍼플립: 모든 코너는 제자리, 모든 엣지는 제자리에서 뒤집힘. 6개 면 정의를 한꺼번에 검증."""
    s = apply_moves(solved_state(), "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2")
    solved = solved_state()
    edge_ids = {_i(a) for a, b in _EDGES} | {_i(b) for a, b in _EDGES}
    for k in range(54):
        if k not in edge_ids:
            assert s[k] == solved[k], f"비-엣지 facelet {k} 가 바뀜"
    for a, b in _EDGES:
        assert s[_i(a)] == solved[_i(b)] and s[_i(b)] == solved[_i(a)], f"엣지 {a}{b} 가 뒤집히지 않음"


def test_checkerboard_pattern():
    s = apply_moves(solved_state(), "U2 D2 F2 B2 L2 R2")
    opp = {0: 3, 3: 0, 1: 4, 4: 1, 2: 5, 5: 2}
    for f in range(6):
        face = s[f * 9:(f + 1) * 9]
        for k in (0, 2, 4, 6, 8):
            assert face[k] == f
        for k in (1, 3, 5, 7):
            assert face[k] == opp[f]
