# src/build_index.py
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from src.ingest import ingest_dir
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def build():
    docs = ingest_dir()
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]

    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (use cosine after normalization)
    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # save index and metadata
    faiss.write_index(index, os.path.join(MODEL_DIR, "faiss.index"))
    with open(os.path.join(MODEL_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"ids": ids, "texts": texts}, f)
    print("Saved index and metadata to models/")

if __name__ == "__main__":
    build()
