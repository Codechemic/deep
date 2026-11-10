"""채점기. 학생 로컬 검증과 교수자 서버 채점에 같은 코드를 쓴다.

사용 예
    python -m cube.evaluate --problems data/public_problems.json --out result.json

동작
  1. 별도 프로세스(worker)에서 solver.solve.Solver 를 한 번 생성한다 (모델 로딩 시간은 채점에서 제외).
  2. 문제를 하나씩 worker 에 보내고, 제한 시간 안에 응답이 없으면 worker 를 죽이고 새로 띄운다.
  3. 받은 해법은 이 프로세스(harness 쪽 시뮬레이터)에서 검증한다. 학생 코드가 보낸 결과를 믿지 않는다.
  4. 구간(tier)별 해결률, 평균 회전 수, 평균 시간을 집계하고 종합 점수를 낸다.
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .cube import MOVE_INDEX, apply_moves, is_solved, is_valid_state
from .scramble import load_problem_set

DEFAULT_CONFIG = {
    "time_limit_sec": 30.0,      # 문제당 제한 시간
    "max_moves": 60,             # 해법 최대 길이
    "grace_sec": 5.0,            # 제한 시간 초과 후 강제 종료까지 여유
    "random_tier_ref_len": 20,   # random 구간의 회전 수 효율 계산 기준 (God's number)
    "tier_weights": {            # 종합 점수 가중치 (합 100)
        "d01-05": 10, "d06-10": 20, "d11-15": 25, "d16-20": 25, "random": 20,
    },
    "length_bonus_share": 0.4,   # 구간 점수 중 회전 수 효율이 차지하는 비율
}


# ----------------------------------------------------------------------------
# worker 프로세스
# ----------------------------------------------------------------------------
def _worker_main(solver_dir: str, weights_dir: str, device: str, inq: mp.Queue, outq: mp.Queue) -> None:
    try:
        sys.path.insert(0, solver_dir)
        mod = importlib.import_module("solver.solve")
        solver = mod.Solver(weights_dir=weights_dir, device=device)
        outq.put(("ready", None))
    except Exception:
        outq.put(("init_error", traceback.format_exc()))
        return
    while True:
        item = inq.get()
        if item is None:
            return
        pid, state, time_limit = item
        t0 = time.perf_counter()
        try:
            moves = solver.solve(list(state), time_limit)
            elapsed = time.perf_counter() - t0
            outq.put(("ok", (pid, moves, elapsed)))
        except Exception:
            elapsed = time.perf_counter() - t0
            outq.put(("error", (pid, traceback.format_exc(), elapsed)))


class SolverRunner:
    """worker 프로세스를 관리하며 문제를 하나씩 풀게 한다."""

    def __init__(self, solver_dir: str, weights_dir: str, device: str, init_timeout: float = 300.0):
        self.solver_dir, self.weights_dir, self.device = solver_dir, weights_dir, device
        self.init_timeout = init_timeout
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self._start()

    def _start(self) -> None:
        self.inq, self.outq = self.ctx.Queue(), self.ctx.Queue()
        self.proc = self.ctx.Process(
            target=_worker_main,
            args=(self.solver_dir, self.weights_dir, self.device, self.inq, self.outq),
            daemon=True,
        )
        self.proc.start()
        t0 = time.perf_counter()
        try:
            status, payload = self.outq.get(timeout=self.init_timeout)
        except Exception:
            self._kill()
            raise RuntimeError(f"Solver 초기화가 {self.init_timeout}초 안에 끝나지 않음")
        self.load_time = time.perf_counter() - t0
        if status != "ready":
            self._kill()
            raise RuntimeError(f"Solver 초기화 실패:\n{payload}")

    def _kill(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(5)
            if self.proc.is_alive():
                self.proc.kill()

    def close(self) -> None:
        try:
            self.inq.put(None)
        except Exception:
            pass
        self._kill()

    def solve(self, pid: str, state: list[int], time_limit: float, grace: float):
        """반환: (status, moves, elapsed, error_text)"""
        self.inq.put((pid, state, time_limit))
        try:
            status, payload = self.outq.get(timeout=time_limit + grace)
        except Exception:
            self._kill()
            self._start()  # 다음 문제를 위해 재시작 (모델 재로딩)
            return "timeout", None, time_limit + grace, None
        if status == "ok":
            _, moves, elapsed = payload
            return "ok", moves, elapsed, None
        _, err, elapsed = payload
        return "error", None, elapsed, err


# ----------------------------------------------------------------------------
# 검증과 집계
# ----------------------------------------------------------------------------
def verify_solution(state: list[int], moves) -> tuple[bool, str]:
    if not isinstance(moves, (list, tuple)):
        return False, "해법이 리스트가 아님"
    if len(moves) == 0 and not is_solved(np.asarray(state, dtype=np.int8)):
        return False, "빈 해법 반환 (풀지 못함)"
    for m in moves:
        if not isinstance(m, str) or m not in MOVE_INDEX:
            return False, f"허용되지 않는 회전: {m!r}"
    final = apply_moves(np.asarray(state, dtype=np.int8), list(moves))
    if not is_solved(final):
        return False, "적용 결과가 완성 상태가 아님"
    return True, ""


def evaluate(problems: list[dict], solver_dir: str, weights_dir: str, device: str,
             config: dict, verbose: bool = True) -> dict:
    cfg = {**DEFAULT_CONFIG, **config}
    runner = SolverRunner(solver_dir, weights_dir, device)
    results = []
    try:
        for i, p in enumerate(problems):
            state = p["state"]
            assert is_valid_state(state), f"문제 {p['problem_id']} 의 상태가 유효하지 않음"
            status, moves, elapsed, err = runner.solve(
                p["problem_id"], state, cfg["time_limit_sec"], cfg["grace_sec"])
            rec = {"problem_id": p["problem_id"], "tier": p["tier"],
                   "scramble_depth": p.get("scramble_depth"),
                   "status": status, "elapsed_sec": round(elapsed, 4),
                   "solved": False, "n_moves": None, "reason": ""}
            if status == "ok":
                ok, reason = verify_solution(state, moves)
                if ok and elapsed > cfg["time_limit_sec"]:
                    ok, reason = False, f"제한 시간 초과 ({elapsed:.1f}s)"
                if ok and len(moves) > cfg["max_moves"]:
                    ok, reason = False, f"해법 길이 초과 ({len(moves)} > {cfg['max_moves']})"
                rec["solved"], rec["reason"] = ok, reason
                rec["n_moves"] = len(moves) if isinstance(moves, (list, tuple)) else None
                rec["moves"] = list(moves) if ok else None
            elif status == "timeout":
                rec["reason"] = "응답 없음 (강제 종료)"
            else:
                rec["reason"] = "예외 발생"
                rec["error"] = err
            results.append(rec)
            if verbose:
                mark = "O" if rec["solved"] else "X"
                print(f"[{i + 1:3d}/{len(problems)}] {p['problem_id']:14s} {mark} "
                      f"{rec['n_moves'] if rec['n_moves'] is not None else '-':>3} moves "
                      f"{rec['elapsed_sec']:7.2f}s {rec['reason']}", flush=True)
    finally:
        runner.close()
    summary = summarize(results, cfg)
    summary["model_load_sec"] = round(runner.load_time, 2)
    return {"config": cfg, "summary": summary, "results": results}


def summarize(results: list[dict], cfg: dict) -> dict:
    tiers: dict[str, dict] = {}
    for r in results:
        t = tiers.setdefault(r["tier"], {"n": 0, "solved": 0, "timeouts": 0, "errors": 0,
                                         "lens": [], "times": [], "effs": []})
        t["n"] += 1
        t["times"].append(r["elapsed_sec"])
        if r["status"] == "timeout":
            t["timeouts"] += 1
        elif r["status"] == "error":
            t["errors"] += 1
        if r["solved"]:
            t["solved"] += 1
            t["lens"].append(r["n_moves"])
            ref = r["scramble_depth"] if r["scramble_depth"] else cfg["random_tier_ref_len"]
            t["effs"].append(min(1.0, ref / max(1, r["n_moves"])))
    out, total = {}, 0.0
    share = cfg["length_bonus_share"]
    for name, t in tiers.items():
        rate = t["solved"] / t["n"] if t["n"] else 0.0
        eff = float(np.mean(t["effs"])) if t["effs"] else 0.0
        w = cfg["tier_weights"].get(name, 0)
        score = w * rate * ((1 - share) + share * eff)
        total += score
        out[name] = {"n": t["n"], "solved": t["solved"], "solve_rate": round(rate, 4),
                     "mean_moves": round(float(np.mean(t["lens"])), 2) if t["lens"] else None,
                     "mean_sec": round(float(np.mean(t["times"])), 3),
                     "timeouts": t["timeouts"], "errors": t["errors"],
                     "length_efficiency": round(eff, 4), "weight": w, "score": round(score, 2)}
    return {"tiers": out, "total_score": round(total, 2),
            "n_problems": len(results), "n_solved": sum(r["solved"] for r in results)}


def print_summary(summary: dict) -> None:
    print("\n구간        문제  해결  해결률   평균회전  평균시간  효율   점수")
    for name, t in summary["tiers"].items():
        mm = f"{t['mean_moves']:.1f}" if t["mean_moves"] is not None else "-"
        print(f"{name:10s} {t['n']:5d} {t['solved']:5d}  {t['solve_rate'] * 100:5.1f}%  "
              f"{mm:>8s}  {t['mean_sec']:7.2f}s  {t['length_efficiency']:.2f}  "
              f"{t['score']:5.1f}/{t['weight']}")
    print(f"\n종합 점수: {summary['total_score']:.2f} / 100   "
          f"(해결 {summary['n_solved']}/{summary['n_problems']}, 모델 로딩 {summary.get('model_load_sec', 0)}s)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="큐브 솔버 채점기")
    ap.add_argument("--problems", required=True, help="문제 세트 JSON")
    ap.add_argument("--solver-dir", default=".", help="solver/ 패키지가 들어 있는 폴더")
    ap.add_argument("--weights-dir", default=None, help="기본값: <solver-dir>/solver/weights")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    ap.add_argument("--config", default=None, help="채점 설정 JSON (기본값 덮어쓰기)")
    ap.add_argument("--limit", type=int, default=None, help="앞에서부터 N문제만 (빠른 확인용)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    solver_dir = str(Path(a.solver_dir).resolve())
    weights_dir = a.weights_dir or str(Path(solver_dir) / "solver" / "weights")
    config = json.loads(Path(a.config).read_text(encoding="utf-8")) if a.config else {}
    problems = load_problem_set(a.problems)
    if a.limit:
        problems = problems[: a.limit]

    report = evaluate(problems, solver_dir, weights_dir, a.device, config, verbose=not a.quiet)
    print_summary(report["summary"])
    if a.out:
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"결과 저장: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
