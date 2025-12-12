#!/usr/bin/env python
"""Heuristic audit of pytest tests.

Goals:
- Flag tests that are likely "happy-path only" (no assertions / very few assertions).
- Flag tests that swallow exceptions (`except: pass`).
- Summarize stress/performance coverage.

This is intentionally heuristic: it produces a shortlist for human review.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ASSERT_CALL_ATTR_PREFIXES = ("assert",)
ASSERT_LIKE_CALLS = {
    # common patterns that indicate an assertion even if not using `assert` statement
    "raises",  # pytest.raises
    "approx",  # pytest.approx in assert expressions (counted separately by AST Assert)
    "fail",  # pytest.fail
}


WEAK_ASSERT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("is_not_none", re.compile(r"^assert\s+.+\s+is\s+not\s+None\s*$")),
    ("isinstance", re.compile(r"^assert\s+isinstance\(.*\)\s*$")),
    ("len_gt0", re.compile(r"^assert\s+len\(.*\)\s*>\s*0\s*$")),
    ("truthy", re.compile(r"^assert\s+.+\s*$")),
]


@dataclass(frozen=True)
class TestFinding:
    file: str
    test_name: str
    asserts: int
    assert_like_calls: int
    stmts: int
    pass_stmts: int
    except_pass_blocks: int
    weak_assert_lines: int
    mock_related_calls: int

    @property
    def total_assert_signals(self) -> int:
        return self.asserts + self.assert_like_calls


def iter_test_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("test_*.py"))


class _TestVisitor(ast.NodeVisitor):
    def __init__(self, source_lines: list[str]):
        self.source_lines = source_lines
        self.current_test: dict | None = None
        self.findings: list[TestFinding] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not node.name.startswith("test"):
            return

        prev = self.current_test
        self.current_test = {
            "name": node.name,
            "asserts": 0,
            "assert_like_calls": 0,
            "mock_related_calls": 0,
            "stmts": 0,
            "pass_stmts": 0,
            "except_pass_blocks": 0,
            "weak_assert_lines": 0,
        }

        # count top-level statements excluding docstring
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
            self.current_test["stmts"] += 1

        self.generic_visit(node)

        f = TestFinding(
            file="",
            test_name=self.current_test["name"],
            asserts=self.current_test["asserts"],
            assert_like_calls=self.current_test["assert_like_calls"],
            stmts=self.current_test["stmts"],
            pass_stmts=self.current_test["pass_stmts"],
            except_pass_blocks=self.current_test["except_pass_blocks"],
            weak_assert_lines=self.current_test["weak_assert_lines"],
            mock_related_calls=self.current_test["mock_related_calls"],
        )
        self.findings.append(f)
        self.current_test = prev

    def visit_Assert(self, node: ast.Assert):
        if self.current_test is not None:
            self.current_test["asserts"] += 1
            # weak assert heuristic via source line
            if getattr(node, "lineno", None):
                line = self.source_lines[node.lineno - 1].strip()
                if any(pat.match(line) for _, pat in WEAK_ASSERT_PATTERNS[:3]):
                    self.current_test["weak_assert_lines"] += 1
        self.generic_visit(node)

    def visit_Pass(self, node: ast.Pass):
        if self.current_test is not None:
            self.current_test["pass_stmts"] += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try):
        # detect `except: pass` or `except Exception: pass`
        if self.current_test is not None:
            for handler in node.handlers:
                if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                    self.current_test["except_pass_blocks"] += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if self.current_test is not None:
            # pytest.raises / pytest.fail etc
            if isinstance(node.func, ast.Attribute) and node.func.attr in ASSERT_LIKE_CALLS:
                self.current_test["assert_like_calls"] += 1
            if isinstance(node.func, ast.Name) and node.func.id in ASSERT_LIKE_CALLS:
                self.current_test["assert_like_calls"] += 1

            # unittest-style self.assert*
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith(ASSERT_CALL_ATTR_PREFIXES):
                self.current_test["assert_like_calls"] += 1

            # mocking / patching signals
            if isinstance(node.func, ast.Name) and node.func.id in {"patch", "MagicMock", "Mock"}:
                self.current_test["mock_related_calls"] += 1
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"patch", "MagicMock", "Mock"}:
                self.current_test["mock_related_calls"] += 1

        self.generic_visit(node)


def scan_file(fp: Path) -> list[TestFinding]:
    try:
        src = fp.read_text(encoding="utf-8")
    except Exception:
        return []

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    lines = src.splitlines()
    visitor = _TestVisitor(lines)
    visitor.visit(tree)

    out = []
    for f in visitor.findings:
        out.append(
            TestFinding(
                file=str(fp).replace("\\", "/"),
                test_name=f.test_name,
                asserts=f.asserts,
                assert_like_calls=f.assert_like_calls,
                stmts=f.stmts,
                pass_stmts=f.pass_stmts,
                except_pass_blocks=f.except_pass_blocks,
                weak_assert_lines=f.weak_assert_lines,
                mock_related_calls=f.mock_related_calls,
            )
        )
    return out


def score(f: TestFinding) -> tuple[int, int, int, int]:
    """Lower score sorts earlier (more suspicious)."""
    # primary: total asserts
    # secondary: many stmts
    # tertiary: swallowed errors
    # quaternary: pass statements
    return (
        f.total_assert_signals,
        -f.stmts,
        -f.except_pass_blocks,
        -f.pass_stmts,
    )


def summarize_markers(test_root: Path) -> dict[str, int]:
    marker_re = re.compile(r"pytest\.mark\.([a-zA-Z_][a-zA-Z0-9_]*)")
    counts: dict[str, int] = {}
    for fp in iter_test_files(test_root):
        try:
            text = fp.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in marker_re.findall(text):
            counts[m] = counts.get(m, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="tests", help="Path to tests root")
    ap.add_argument("--limit", type=int, default=60, help="Max suspicious tests to print")
    ap.add_argument("--write", default="", help="Write a markdown report to this path")
    args = ap.parse_args()

    test_root = Path(args.tests)
    files = list(iter_test_files(test_root))

    all_findings: list[TestFinding] = []
    for fp in files:
        all_findings.extend(scan_file(fp))

    # suspicious: no assertion signals OR 1 signal with many statements OR swallowed errors
    suspicious = [
        f
        for f in all_findings
        if f.total_assert_signals == 0
        or (f.total_assert_signals <= 1 and f.stmts >= 8)
        or f.except_pass_blocks > 0
    ]
    suspicious.sort(key=score)

    stress_files = sorted((test_root / "stress").glob("test_*.py"))
    perf_files = sorted((test_root / "performance").rglob("test_*.py"))
    markers = summarize_markers(test_root)

    lines: list[str] = []
    lines.append(f"Scanned files: {len(files)}")
    lines.append(f"Discovered test functions: {len(all_findings)}")
    lines.append(f"Suspicious tests: {len(suspicious)}")
    lines.append("")
    lines.append("Top suspicious tests (file | test | asserts+signals | stmts | except:pass | pass | mock_calls):")
    for f in suspicious[: args.limit]:
        lines.append(
            f"- {f.file} | {f.test_name} | {f.total_assert_signals} | {f.stmts} | {f.except_pass_blocks} | {f.pass_stmts} | {f.mock_related_calls}"
        )

    lines.append("")
    lines.append("Stress coverage:")
    if stress_files:
        for fp in stress_files:
            lines.append(f"- {str(fp).replace('\\\\','/')}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("Performance coverage:")
    if perf_files:
        lines.append(f"- test files: {len(perf_files)}")
        # list a few
        for fp in perf_files[:10]:
            lines.append(f"- {str(fp).replace('\\\\','/')}")
        if len(perf_files) > 10:
            lines.append(f"- ... ({len(perf_files) - 10} more)")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("Pytest marker usage (approx by text scan):")
    for k, v in list(markers.items())[:25]:
        lines.append(f"- {k}: {v}")

    report = "\n".join(lines) + "\n"
    print(report)

    if args.write:
        out_path = Path(args.write)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
