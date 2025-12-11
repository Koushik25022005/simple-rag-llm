# src/query.py
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai
import numpy as np
import sys

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", 5))
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# initialize models
embed_model = SentenceTransformer(EMBEDDING_MODEL)
faiss_index = faiss.read_index(os.path.join(MODEL_DIR, "faiss.index"))
with open(os.path.join(MODEL_DIR, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)

def retrieve(query: str, top_k: int = TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = faiss_index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        results.append({"id": meta["ids"][idx], "text": meta["texts"][idx]})
    return results

def answer_with_openai(query: str, contexts):
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set for LLM generation.")
    openai.api_key = OPENAI_KEY
    prompt = "You are a helpful assistant. Use the following context to answer the question.\n\n"
    for i, c in enumerate(contexts):
        prompt += f"Context {i+1}: {c['text']}\n\n"
    prompt += f"Question: {query}\nAnswer:"
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # small example; replace with your available model
        messages=[{"role":"user","content":prompt}],
        max_tokens=300,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"].strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/query.py \"your question\"")
        sys.exit(1)
    q = sys.argv[1]
    ctxs = retrieve(q)
    print("Retrieved contexts:")
    for r in ctxs:
        print("-", r["id"], ":", r["text"][:200].replace("\n", " "))
    if OPENAI_KEY:
        print("\nGenerating answer via OpenAI...")
        ans = answer_with_openai(q, ctxs)
        print("\nAnswer:\n", ans)
    else:
        print("\nNo OPENAI_API_KEY found. Returning retrieved contexts only.")
