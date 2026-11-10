"""제출 전 자가 검사. 푸시하기 전에 반드시 실행할 것.

    python scripts/check_submission.py

검사 항목
  1. solver/solve.py 에 Solver 클래스가 있고 생성 가능한가
  2. solver/weights/ 가 존재하고 용량 제한 이내인가
  3. solver/ 안에 금지 패키지 import 가 없는가
  4. 공개 문제 세트의 앞 몇 문제를 실제로 풀어 보는가 (인터페이스 위반 탐지)
  5. REPORT.md 가 있는가
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_WEIGHTS_MB = 100
BANNED_MODULES = {"kociemba", "rubik", "rubik_solver", "pycuber", "twophase", "RubikTwoPhase",
                  "requests", "urllib", "socket", "http", "subprocess", "aiohttp", "httpx"}
ALLOWED_TOP_LEVEL = {"torch", "numpy", "cube", "solver"}


def fail(msg: str) -> None:
    print(f"  [실패] {msg}")
    fail.count += 1


fail.count = 0


def ok(msg: str) -> None:
    print(f"  [통과] {msg}")


def check_files() -> None:
    print("1. 파일 구조")
    for rel in ["solver/__init__.py", "solver/solve.py", "REPORT.md"]:
        if (ROOT / rel).exists():
            ok(rel)
        else:
            fail(f"{rel} 없음")
    wdir = ROOT / "solver" / "weights"
    if not wdir.is_dir():
        fail("solver/weights/ 폴더 없음")
        return
    files = [p for p in wdir.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    if not files:
        fail("solver/weights/ 가 비어 있음")
    elif size_mb > MAX_WEIGHTS_MB:
        fail(f"가중치 용량 {size_mb:.1f} MB > {MAX_WEIGHTS_MB} MB")
    else:
        ok(f"가중치 {len(files)}개, {size_mb:.1f} MB")


def check_imports() -> None:
    print("2. import 검사 (solver/ 하위 .py)")
    import_std = set(sys.stdlib_module_names)
    for py in (ROOT / "solver").rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError as e:
            fail(f"{py.relative_to(ROOT)} 문법 오류: {e}")
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top in BANNED_MODULES:
                    fail(f"{py.relative_to(ROOT)}: 금지 모듈 import '{name}'")
                elif top not in ALLOWED_TOP_LEVEL and top not in import_std:
                    fail(f"{py.relative_to(ROOT)}: 허용되지 않은 패키지 '{name}' (채점 서버에 없음)")
    if fail.count == 0:
        ok("금지/비허용 import 없음")


def check_notebooks() -> None:
    print("3. 노트북 파일")
    nbs = [p for p in (ROOT / "solver").rglob("*.ipynb")]
    if nbs:
        fail(f"solver/ 안에 .ipynb 가 있음 ({len(nbs)}개). 모듈(.py)만 제출")
    else:
        ok("solver/ 에 노트북 없음")


def check_run(limit: int) -> None:
    print(f"4. 공개 문제 {limit}개 실행")
    cmd = [sys.executable, "-m", "cube.evaluate", "--problems", "data/public_problems.json",
           "--solver-dir", str(ROOT), "--limit", str(limit), "--quiet", "--out", ".check_result.json"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        fail("채점기 실행 실패\n" + r.stdout[-2000:] + r.stderr[-2000:])
        return
    rep = json.loads((ROOT / ".check_result.json").read_text(encoding="utf-8"))
    errs = [x for x in rep["results"] if x["status"] == "error"]
    if errs:
        fail(f"{len(errs)}개 문제에서 예외 발생. 첫 예외:\n{errs[0].get('error', '')[-1500:]}")
    else:
        ok(f"예외 없이 실행됨. 해결 {rep['summary']['n_solved']}/{rep['summary']['n_problems']}, "
           f"모델 로딩 {rep['summary']['model_load_sec']}s")
    (ROOT / ".check_result.json").unlink(missing_ok=True)


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    check_files()
    check_imports()
    check_notebooks()
    if fail.count == 0:
        check_run(limit)
    else:
        print("4. 앞 단계 실패로 실행 검사 생략")
    print("\n결과:", "모두 통과" if fail.count == 0 else f"실패 {fail.count}건")
    return 0 if fail.count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
