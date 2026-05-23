from __future__ import annotations
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from code_rag.embed import index_repo
from code_rag.retrieve import load_retriever, retrieve
from code_rag.rerank import load_reranker, rerank
from code_rag.generate import generate

console = Console()


def cmd_index(args: argparse.Namespace) -> None:
    repo_path = Path(args.path)
    if not repo_path.exists():
        console.print(f"[red]路径不存在: {repo_path}[/red]")
        sys.exit(1)
    index_repo(repo_path)


def cmd_ask(args: argparse.Namespace) -> None:
    query = args.query
    console.print(f"[bold]查询：[/bold] {query}\n")

    with console.status("向量检索中..."):
        model, collection = load_retriever()
        hits = retrieve(query, model, collection, top_k=10)

    with console.status("重排序中..."):
        reranker = load_reranker()
        top_chunks = rerank(query, hits, reranker, top_k=3)

    console.print("[bold cyan]=== 引用的代码 ===[/bold cyan]")
    for c in top_chunks:
        console.print(
            f"  [yellow]{c['file_path']}:{c['start_line']}-{c['end_line']}[/yellow]"
            f" [{c['kind']}] [bold]{c['qualified_name']}[/bold]"
            f"  score={c['rerank_score']}"
        )

    with console.status("LLM 生成回答中..."):
        answer = generate(query, top_chunks)

    console.print()
    console.print(Panel(
        Markdown(answer),
        title="[bold green]回答[/bold green]",
        border_style="green",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-rag",
        description="用自然语言问 Python repo 的代码问题",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="索引一个 Python repo")
    p_index.add_argument("path", nargs="?", default=".", help="repo 路径")

    p_ask = sub.add_parser("ask", help="问一个关于代码的问题")
    p_ask.add_argument("query", nargs="+", help="你的问题")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "ask":
        args.query = " ".join(args.query)
        cmd_ask(args)


if __name__ == "__main__":
    main()