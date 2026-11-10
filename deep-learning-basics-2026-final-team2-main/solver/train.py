# solver/train.py
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cube.cube import MOVE_PERMS, MOVES, one_hot, solved_state
from solver.model import CubeNet

_FACE_OF_MOVE = np.array([MOVES.index(m) // 3 for m in MOVES])


def sample_batch(n: int, max_depth: int, rng: np.random.Generator):
    """(상태, 섞은 횟수) 배치 생성. 같은 면 연속 회전은 피한다."""
    depth = rng.integers(0, max_depth + 1, size=n)
    states = np.tile(solved_state(), (n, 1))
    last_face = np.full(n, -1)
    rows = np.arange(n)
    for t in range(max_depth):
        mv = rng.integers(0, len(MOVES), size=n)
        same = _FACE_OF_MOVE[mv] == last_face
        mv = np.where(same, (mv + 3) % len(MOVES), mv)
        active = t < depth
        new_states = states[rows[:, None], MOVE_PERMS[mv]]
        states = np.where(active[:, None], new_states, states)
        last_face = np.where(active, _FACE_OF_MOVE[mv], last_face)
    return states.astype(np.int8), depth


def make_val_set(n: int, max_depth: int, seed: int):
    """고정 검증 세트 생성."""
    rng = np.random.default_rng(seed)
    return sample_batch(n, max_depth, rng)


def load_policy_dataset(path: Path):
    """policy_dataset.npz 로딩 → (x_onehot, policy_index) 반환."""
    d = np.load(path)
    x = d["x"]            # (N, 324)
    policy = d["policy"]  # (N,) int64, MOVES index
    return x, policy


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000,
                    help="최대 스텝 수 (조기 종료로 더 일찍 끝날 수 있음)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-depth", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="solver/weights/model.pt")
    ap.add_argument("--seed", type=int, default=0)
    # ---- 조기 종료 / checkpoint ----
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--val-seed", type=int, default=12345)
    # ---- 정책 데이터 ----
    ap.add_argument("--policy-data", default=None,
                    help="정책 라벨 npz 경로 (예: solver/data/policy_dataset.npz)")
    ap.add_argument("--policy-batch", type=int, default=256,
                    help="정책 데이터에서 뽑을 배치 크기")
    ap.add_argument("--policy-weight", type=float, default=1.0,
                    help="정책 손실 가중치 (거리 loss 대비)")
    ap.add_argument("--policy-sample-rate", type=float, default=0.5,
                    help="각 스텝에서 정책 데이터를 섞을 확률")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    # ---- 고정 검증 세트 ----
    val_states, val_depth = make_val_set(a.val_n, a.max_depth, a.val_seed)
    x_val = torch.from_numpy(one_hot(val_states)).to(a.device)
    y_val = torch.from_numpy(val_depth).long().to(a.device)

    # ---- 정책 데이터 (있으면) ----
    policy_x = None
    policy_y = None
    if a.policy_data:
        px, py = load_policy_dataset(Path(a.policy_data))
        policy_x = torch.from_numpy(px).to(a.device)   # (P, 324)
        policy_y = torch.from_numpy(py).long().to(a.device)  # (P,)
        print(f"정책 데이터 로드: {policy_x.shape[0]} 샘플  ({a.policy_data})", flush=True)

    model = CubeNet(n_classes=a.max_depth + 1,
                    hidden=a.hidden,
                    n_layers=a.layers,
                    dropout=a.dropout).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    best_step = 0
    no_improve = 0

    t0 = time.time()
    model.train()

    for step in range(1, a.steps + 1):
        # --- 거리 샘플 (기본) ---
        states, depth = sample_batch(a.batch, a.max_depth, rng)
        x = torch.from_numpy(one_hot(states)).to(a.device)
        y = torch.from_numpy(depth).long().to(a.device)

        # --- 정책 데이터 섞기 ---
        if policy_x is not None and rng.random() < a.policy_sample_rate:
            idx = rng.integers(0, policy_x.shape[0], size=a.policy_batch)
            x_policy = policy_x[idx]
            y_policy = policy_y[idx]            # (B,) MOVES index
            # 정책 데이터로 forward (거리는 라벨 없으므로 거리 loss 없음)
            dist_logits_p, policy_logits_p = model(x_policy)
            policy_loss = F.cross_entropy(policy_logits_p, y_policy)
            # 거리 데이터도 같이 forward
            dist_logits, _ = model(x)
            dist_loss = F.cross_entropy(dist_logits, y)
            loss = dist_loss + a.policy_weight * policy_loss
        else:
            dist_logits, _ = model(x)
            dist_loss = F.cross_entropy(dist_logits, y)
            loss = dist_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ---- 검증 평가 ----
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(x_val)
                val_loss = F.cross_entropy(val_logits, y_val).item()
                val_mae = (model.expected_distance(x_val) - y_val.float()).abs().mean().item()
            model.train()

            improved = val_mae < best_val_mae
            if improved:
                best_val_mae = val_mae
                best_step = step
                no_improve = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": {
                        "n_classes": a.max_depth + 1,
                        "hidden": a.hidden,
                        "n_layers": a.layers,
                        "dropout": a.dropout,
                    },
                }, out)
                print(f"  [checkpoint] step {step:6d}  "
                      f"val_loss {val_loss:.4f}  val_MAE {val_mae:.3f}  "
                      f"(최고 갱신 → 저장)", flush=True)
            else:
                no_improve += 1
                print(f"  [eval] step {step:6d}  "
                      f"val_loss {val_loss:.4f}  val_MAE {val_mae:.3f}  "
                      f"(best {best_val_mae:.3f} @ step {best_step}, "
                      f"no_improve {no_improve}/{a.patience})",
                      flush=True)

            if no_improve >= a.patience:
                print(f"\n조기 종료: {a.patience}회 연속 개선 없음 "
                      f"(최고 val_MAE {best_val_mae:.3f} @ step {best_step})",
                      flush=True)
                break

    elapsed = time.time() - t0
    print(f"\n학습 종료: step {step} / {a.steps}  {elapsed:.0f}s")
    print(f"최고 검증 MAE: {best_val_mae:.3f} @ step {best_step}  "
          f"(저장됨: {out})")


if __name__ == "__main__":
    main()
