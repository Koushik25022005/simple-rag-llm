# src/streamlit_app.py
import sys
from pathlib import Path
# Ensure repo root is on sys.path when running from streamlit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import openai
from src.ingest import ingest_dir
from src.build_index import MODELS_DIR
from src.build_index import build_index as build_index_fn
from src.build_index import build_index
from src.utils import chunk_text
from pathlib import Path
from typing import List
import numpy as np

load_dotenv()

# UI config
st.set_page_config(page_title="RAG Streamlit", layout="centered")
st.title("RAG - LLM Bot")
st.write("Built by Kousik Sripathi Panditaradhyula")
st.write("Upload files to `data/uploads/`, build index, ask questions. Works with PDFs, scanned PDFs, images and text.")

# Sidebar config
embedding_model = st.sidebar.text_input("Embedding model", value=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
top_k = st.sidebar.slider("Top K", 1, 10, int(os.getenv("TOP_K", 1)))
use_openai = st.sidebar.checkbox("Use OpenAI for final answer", value=bool(os.getenv("OPENAI_API_KEY")))
openai_key_input = st.sidebar.text_input("OpenAI API key (optional)", type="password", value="")
openai_api_key = openai_key_input or os.getenv("OPENAI_API_KEY", "")

DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_embed_model(name: str):
    return SentenceTransformer(name)

def save_uploaded_file(uploaded_file):
    out = UPLOADS_DIR / uploaded_file.name
    with open(out, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out

def load_index_from_disk():
    idx_path = MODELS_DIR / "faiss.index"
    meta_path = MODELS_DIR / "meta.pkl"
    if not idx_path.exists() or not meta_path.exists():
        return None, None
    try:
        idx = faiss.read_index(str(idx_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return idx, meta
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return None, None

def retrieve(index, meta, model, query: str, top_k: int):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for rank, idx in enumerate(I[0]):
        results.append({
            "rank": rank+1,
            "score": float(D[0][rank]),
            "id": meta["ids"][idx],
            "text": meta["texts"][idx],
            "meta": meta.get("metadatas", [None]*len(meta["ids"]))[idx]
        })
    return results

def generate_openai_answer(key: str, query: str, contexts: List[dict], model_name="gpt-3.5-turbo-1106"):
    openai.api_key = key
    prompt = "You are a helpful assistant. Use the following contexts to answer the question.\n\n"
    for i, c in enumerate(contexts):
        prompt += f"Context {i+1} (source: {c['meta'].get('source') if c.get('meta') else 'unknown'}):\n{c['text']}\n\n"
    prompt += f"Question: {query}\nAnswer:"
    resp = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        max_tokens=400,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"].strip()

# ---- UI ----

st.header("1) Upload files (optional)")
uploaded = st.file_uploader("Upload .txt / .pdf / images (multi)", accept_multiple_files=True)
if uploaded:
    saved = []
    for u in uploaded:
        p = save_uploaded_file(u)
        saved.append(p.name)
    st.success(f"Saved {len(saved)} files to data/uploads/: {', '.join(saved)}")

st.header("2) Build / Load index")
c1, c2 = st.columns(2)
with c1:
    if st.button("Build index (ingest data/ + uploads)"):
        with st.spinner("Building index — this may take a minute..."):
            build_index_fn(embedding_model_name=embedding_model)
            st.success("Index built and saved.")
with c2:
    if st.button("Load index from disk"):
        idx_meta = load_index_from_disk()
        if idx_meta[0] is not None:
            st.session_state["faiss_index"], st.session_state["meta"] = idx_meta
            st.success("Loaded index into session.")

if "faiss_index" in st.session_state:
    st.info(f"Index in session — {len(st.session_state['meta']['ids'])} chunks indexed.")
else:
    st.warning("No index in session. Build or Load one.")

st.header("3) Ask a question")
query = st.text_input("Enter your question here")
if st.button("Ask") and query.strip():
    if "faiss_index" not in st.session_state:
        # attempt to auto-load from disk
        idx_meta = load_index_from_disk()
        if idx_meta[0] is None:
            st.error("No index available. Build or Load an index first.")
            st.stop()
        st.session_state["faiss_index"], st.session_state["meta"] = idx_meta

    model = st.session_state.get("embed_model") or load_embed_model(embedding_model)
    st.session_state["embed_model"] = model
    index = st.session_state["faiss_index"]
    meta = st.session_state["meta"]

    with st.spinner("Retrieving..."):
        contexts = retrieve(index, meta, model, query, top_k=top_k)

    st.subheader("Retrieved contexts")
    for c in contexts:
        src = c["meta"].get("source") if c.get("meta") else "unknown"
        scanned = c["meta"].get("is_scanned_pdf", False) if c.get("meta") else False
        badge = " (scanned PDF OCR)" if scanned else ""
        st.markdown(f"**{c['rank']}. {src}{badge} — score {c['score']:.3f}**")
        st.write(c["text"][:1200])

    if use_openai and openai_api_key:
        with st.spinner("Generating answer with OpenAI..."):
            try:
                ans = generate_openai_answer(openai_api_key, query, contexts)
                st.subheader("OpenAI answer")
                st.write(ans)
            except Exception as e:
                st.error(f"OpenAI request failed: {e}")
    else:
        st.info("OpenAI disabled or key not provided — showing contexts only.")
