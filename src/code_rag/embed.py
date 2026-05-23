from __future__ import annotations
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

try:
    from .parse import CodeChunk, parse_repo
except ImportError:
    from parse import CodeChunk, parse_repo

MODEL_NAME  = "BAAI/bge-base-en-v1.5"
CHROMA_PATH = ".chroma"
COLLECTION  = "code_chunks"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> SentenceTransformer:
    device = _device()
    print(f"加载 {MODEL_NAME}  (device={device}) ...")
    return SentenceTransformer(MODEL_NAME, device=device)


def get_collection(chroma_path: str = CHROMA_PATH, name: str = COLLECTION) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def embed_and_store(
    chunks: list[CodeChunk],
    model: SentenceTransformer,
    collection: chromadb.Collection,
    batch_size: int = 32,
) -> None:
    texts = [c.code for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    ids, metadatas = [], []
    for c in chunks:
        path = c.file_path.replace("\\", "/")
        ids.append(f"{path}:{c.start_line}:{c.qualified_name}")
        metadatas.append({
            "file_path": path,
            "qualified_name": c.qualified_name,
            "kind": c.kind,
            "start_line": c.start_line,
            "end_line": c.end_line,
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )


def index_repo(repo_root: Path, chroma_path: str = CHROMA_PATH, collection_name: str = COLLECTION) -> int:
    print(f"正在索引 repo: {repo_root.resolve()}")
    chunks = parse_repo(repo_root)
    if not chunks:
        print("没有找到任何 chunk，退出")
        return 0

    model      = load_model()
    collection = get_collection(chroma_path, name=collection_name)

    print(f"embedding {len(chunks)} 个 chunk ...")
    embed_and_store(chunks, model, collection)
    print(f"完成，已写入 ChromaDB ({len(chunks)} 个 chunk，collection={collection_name})")
    return len(chunks)


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    index_repo(root)