# src/build_index.py
import os
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from src.ingest import ingest_dir, ROOT
import numpy as np
from tqdm import tqdm

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def build_index(embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                batch_size: int = 32,
                normalize: bool = True):
    docs = ingest_dir()
    if not docs:
        print("No documents to index. Add files to data/ and try again.")
        return

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metadatas = [d["meta"] for d in docs]

    model = SentenceTransformer(embedding_model_name)

    # compute embeddings in batches to reduce peak memory
    embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i+batch_size]
        batch_emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        embeds.append(batch_emb)
    embeddings = np.vstack(embeds)

    if normalize:
        faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index + meta
    faiss.write_index(index, str(MODELS_DIR / "faiss.index"))
    with open(MODELS_DIR / "meta.pkl", "wb") as f:
        pickle.dump({"ids": ids, "texts": texts, "metadatas": metadatas}, f)
    print(f"Saved FAISS index and metadata to {MODELS_DIR}")

if __name__ == "__main__":
    build_index()
