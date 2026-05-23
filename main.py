from __future__ import annotations
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from code_rag.embed import index_repo, CHROMA_PATH, COLLECTION
from code_rag.retrieve import load_retriever, retrieve
from code_rag.rerank import load_reranker, rerank
from code_rag.generate import generate

console = Console()


def cmd_index(args: argparse.Namespace) -> None:
    repo_path = Path(args.path)
    if not repo_path.exists():
        console.print(f"[red]路径不存在: {repo_path}[/red]")
        sys.exit(1)
    collection_name = args.collection or repo_path.resolve().name
    index_repo(repo_path, collection_name=collection_name)


def cmd_ask(args: argparse.Namespace) -> None:
    query = args.query
    collection_name = args.collection or COLLECTION
    console.print(f"[bold]查询：[/bold] {query}  [dim](collection={collection_name})[/dim]\n")

    with console.status("向量检索中..."):
        model, collection = load_retriever(collection_name=collection_name)
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


def cmd_list(args: argparse.Namespace) -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collections = client.list_collections()
    if not collections:
        console.print("[yellow]没有任何已索引的 repo[/yellow]")
        return
    console.print("[bold cyan]已索引的 repo：[/bold cyan]")
    for col in collections:
        console.print(f"  [green]{col.name}[/green]  ({col.count()} 个 chunk)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-rag",
        description="用自然语言问 Python repo 的代码问题",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="索引一个 Python repo")
    p_index.add_argument("path", nargs="?", default=".", help="repo 路径")
    p_index.add_argument("--collection", default=None, help="collection 名（默认用 repo 文件夹名）")

    p_ask = sub.add_parser("ask", help="问一个关于代码的问题")
    p_ask.add_argument("query", nargs="+", help="你的问题")
    p_ask.add_argument("--collection", default=None, help="指定查哪个 repo（默认 code_chunks）")

    sub.add_parser("list", help="列出所有已索引的 repo")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "ask":
        args.query = " ".join(args.query)
        cmd_ask(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()