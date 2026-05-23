from __future__ import annotations
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()
try:
    from .retrieve import load_retriever, retrieve
    from .rerank import load_reranker, rerank
except ImportError:
    from retrieve import load_retriever, retrieve
    from rerank import load_reranker, rerank

# DeepSeek V4 Flash via Anthropic SDK
client = anthropic.Anthropic(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/anthropic",
)
MODEL = "deepseek-v4-flash"

SYSTEM_PROMPT = """You are a code assistant. Answer questions about a Python codebase.

Rules:
- Answer ONLY based on the provided code context.
- Always cite file paths and line numbers, e.g. `src/parse.py:28`.
- If the context lacks enough information, say so clearly.
- Be concise and precise."""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        header = (
            f"# {c['file_path']}:{c['start_line']}-{c['end_line']}"
            f" [{c['kind']}] {c['qualified_name']}"
        )
        parts.append(f"{header}\n{c['code']}")
    return "\n\n---\n\n".join(parts)


def generate(query: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    user_message = f"Code context:\n\n{context}\n\n---\n\nQuestion: {query}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return next(block.text for block in response.content if hasattr(block, "text"))


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "how does the AST parser work"
    print(f"查询：{query}\n")

    model, collection = load_retriever()
    hits = retrieve(query, model, collection, top_k=10)

    reranker = load_reranker()
    top_chunks = rerank(query, hits, reranker, top_k=3)

    print("=== 引用的代码 ===")
    for c in top_chunks:
        print(f"  {c['file_path']}:{c['start_line']} [{c['kind']}] {c['qualified_name']}")

    print("\n=== LLM 回答 ===")
    answer = generate(query, top_chunks)
    print(answer)