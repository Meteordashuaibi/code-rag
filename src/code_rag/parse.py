from __future__ import annotations
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class CodeChunk:
    file_path: str       # 相对 repo 根的路径，例如 "src/utils.py"
    qualified_name: str  # 这个单元的名字，例如 "load_config"
    kind: str            # "function" / "async_function" / "class"
    start_line: int      # 起始行号（从 1 开始）
    end_line: int        # 结束行号（含）
    code: str            # 这段的源码文本


SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"}


def iter_python_files(repo_root: Path) -> Iterator[Path]:
    for path in repo_root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def extract_chunks(source: str, file_path: str) -> list[CodeChunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    chunks = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        kind = (
            "class"          if isinstance(node, ast.ClassDef)          else
            "async_function" if isinstance(node, ast.AsyncFunctionDef)  else
            "function"
        )
        start = node.lineno
        end   = node.end_lineno
        code  = "\n".join(lines[start - 1 : end])

        chunks.append(CodeChunk(
            file_path=file_path,
            qualified_name=node.name,
            kind=kind,
            start_line=start,
            end_line=end,
            code=code,
        ))

    return chunks


def parse_repo(repo_root: Path) -> list[CodeChunk]:
    all_chunks: list[CodeChunk] = []
    for py_file in iter_python_files(repo_root):
        source = py_file.read_text(encoding="utf-8", errors="ignore")
        rel_path = str(py_file.relative_to(repo_root))
        all_chunks.extend(extract_chunks(source, rel_path))
    return all_chunks


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    chunks = parse_repo(root)
    print(f"找到 {len(chunks)} 个 chunk")
    for c in chunks[:10]:
        print(f"  {c.file_path}:{c.start_line}-{c.end_line} [{c.kind}] {c.qualified_name}")