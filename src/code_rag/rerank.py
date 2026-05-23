from __future__ import annotations
from sentence_transformers import CrossEncoder

RERANKER_NAME = "BAAI/bge-reranker-v2-m3"


def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_NAME)


def rerank(
    query: str,
    hits: list[dict],
    reranker: CrossEncoder,
    top_k: int = 3,
) -> list[dict]:
    if not hits:
        return []

    pairs = [(query, h["code"]) for h in hits]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(hits, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {**hit, "rerank_score": round(float(score), 4)}
        for hit, score in ranked[:top_k]
    ]


if __name__ == "__main__":
    import sys
    from retrieve import load_retriever, retrieve

    query = " ".join(sys.argv[1:]) or "how does parsing work"
    print(f"查询：{query}\n")

    model, collection = load_retriever()
    hits = retrieve(query, model, collection, top_k=10)

    reranker = load_reranker()
    results = rerank(query, hits, reranker)

    print("=== Rerank 结果 ===")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['file_path']}:{r['start_line']}-{r['end_line']} "
              f"[{r['kind']}] {r['qualified_name']}  rerank_score={r['rerank_score']}")
        print(f"     {r['code'][:80].strip()} ...")
        print()