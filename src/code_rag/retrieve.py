from __future__ import annotations
import torch
from sentence_transformers import SentenceTransformer
import chromadb

try:
    from .embed import MODEL_NAME, CHROMA_PATH, COLLECTION, _device
except ImportError:
    from embed import MODEL_NAME, CHROMA_PATH, COLLECTION, _device

def load_retriever() -> tuple[SentenceTransformer, chromadb.Collection]:
    model = SentenceTransformer(MODEL_NAME, device=_device())
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION)
    return model, collection


def retrieve(
    query: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    top_k: int = 5,
) -> list[dict]:
    query_vec = model.encode(query, convert_to_numpy=True)

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "file_path": meta["file_path"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "qualified_name": meta["qualified_name"],
            "kind": meta["kind"],
            "score": round(1 - dist, 4),   # cosine distance → similarity
            "code": doc,
        })

    return hits


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "how does parsing work"
    print(f"查询：{query}\n")

    model, collection = load_retriever()
    hits = retrieve(query, model, collection)

    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['file_path']}:{h['start_line']}-{h['end_line']} "
              f"[{h['kind']}] {h['qualified_name']}  score={h['score']}")
        print(f"     {h['code'][:80].strip()} ...")
        print()