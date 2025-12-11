# src/app.py
import os
import tempfile
import pickle
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import openai
from src.utils import load_any, chunk_text, get_ext
from src.ingest import ingest_dir
from pathlib import Path
import numpy as np

load_dotenv()

# Config / paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = ROOT / "models"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_embed_model(model_name: str):
    return SentenceTransformer(model_name)

def save_uploaded_file(uploaded_file):
    out_path = UPLOADS_DIR / uploaded_file.name
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(out_path)

def build_index_from_dir(embedding_model_name: str, normalize: bool = True):
    """
    Build FAISS index from files in data/ and data/uploads/.
    Returns (index, meta_dict, model) where meta_dict = {"ids": [...], "texts": [...]}
    """
    st.info("Ingesting files from data/ and data/uploads/ ...")
    docs = ingest_dir()  # uses data/ (includes uploads subdir content)
    if len(docs) == 0:
        st.warning("No documents found in data/ — upload files first or add documents to data/")
        return None, None, None

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]

    st.info("Loading embedding model...")
    model = load_embed_model(embedding_model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    if normalize:
        faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # use inner product (cosine if normalized)
    index.add(embeddings)

    meta = {"ids": ids, "texts": texts}
    # Save to disk for later reuse
    faiss.write_index(index, str(MODELS_DIR / "faiss.index"))
    with open(MODELS_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    st.success(f"Built index with {len(ids)} chunks and saved to {MODELS_DIR}/")
    return index, meta, model

def load_index_from_disk():
    idx_path = MODELS_DIR / "faiss.index"
    meta_path = MODELS_DIR / "meta.pkl"
    if not idx_path.exists() or not meta_path.exists():
        st.warning("No saved index found on disk.")
        return None, None
    index = faiss.read_index(str(idx_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def retrieve_from_index(index, meta, model, query: str, top_k: int = 5):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        results.append({"id": meta["ids"][idx], "text": meta["texts"][idx]})
    return results

def generate_openai_answer(openai_api_key: str, query: str, contexts, model_name="gpt-3.5-turbo"):
    openai.api_key = openai_api_key
    prompt = "You are a helpful assistant. Use the following context to answer the question.\n\n"
    for i, c in enumerate(contexts):
        prompt += f"Context {i+1}: {c['text']}\n\n"
    prompt += f"Question: {query}\nAnswer:"
    resp = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"].strip()

# ---- Streamlit UI ----
st.set_page_config(page_title="Simple RAG (Streamlit)", layout="centered")
st.title("Simple RAG — Streamlit UI")
st.markdown("Upload `.txt`, `.pdf`, or image files; build an index; then ask natural questions.")

# Sidebar - settings
st.sidebar.header("Settings")
embedding_model = st.sidebar.text_input("Embedding model", value=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
top_k = st.sidebar.number_input("Top K (retrieval)", min_value=1, max_value=20, value=int(os.getenv("TOP_K", 5)))
use_openai = st.sidebar.checkbox("Use OpenAI to generate answer", value=bool(os.getenv("OPENAI_API_KEY")))
openai_key_input = st.sidebar.text_input("OpenAI API key (optional, overrides .env)", type="password", value="")
openai_api_key = openai_key_input or os.getenv("OPENAI_API_KEY", "")

st.sidebar.markdown("**Notes**\n- Install Tesseract on your machine for OCR to work.\n- Uploaded files are saved to `data/uploads/`.")

# File uploader
st.header("1) Upload files (optional)")
uploaded = st.file_uploader("Upload one or more files (.txt, .pdf, images)", accept_multiple_files=True)
if uploaded:
    saved = []
    for f in uploaded:
        path = save_uploaded_file(f)
        saved.append(path)
    st.success(f"Saved {len(saved)} uploaded files to `data/uploads/`.")
    st.write([os.path.basename(p) for p in saved])

# Index controls
st.header("2) Build / Load index")
col1, col2 = st.columns(2)
with col1:
    if st.button("Build index from data/ (includes uploaded files)"):
        with st.spinner("Building index — this may take a minute..."):
            idx, meta, model = build_index_from_dir(embedding_model)
            if idx is not None:
                st.session_state["faiss_index"] = idx
                st.session_state["meta"] = meta
                st.session_state["embed_model"] = model
with col2:
    if st.button("Load saved index from disk"):
        idx_meta = load_index_from_disk()
        if idx_meta[0] is not None:
            st.session_state["faiss_index"], st.session_state["meta"] = idx_meta
            st.success("Loaded index and metadata into session.")

# Show index status
if "faiss_index" in st.session_state:
    st.info(f"Index loaded in session. Chunks indexed: {len(st.session_state['meta']['ids'])}")
else:
    st.warning("No index loaded in session. Build or load one to query.")

# Query area
st.header("3) Ask a question")
query = st.text_input("Your question", "")
if st.button("Ask") and query.strip():
    if "faiss_index" not in st.session_state:
        st.error("No index loaded. Build or load an index first.")
    else:
        embed_model = st.session_state.get("embed_model") or load_embed_model(embedding_model)
        index = st.session_state["faiss_index"]
        meta = st.session_state["meta"]

        with st.spinner("Retrieving relevant chunks..."):
            contexts = retrieve_from_index(index, meta, embed_model, query, top_k=top_k)

        st.subheader("Retrieved contexts")
        for i, c in enumerate(contexts):
            st.markdown(f"**Context {i+1} — {c['id']}**")
            st.write(c["text"][:1200])

        if use_openai and (openai_api_key):
            st.subheader("Generated answer (OpenAI)")
            try:
                with st.spinner("Calling OpenAI..."):
                    ans = generate_openai_answer(openai_api_key, query, contexts)
                st.write(ans)
            except Exception as e:
                st.error(f"OpenAI call failed: {e}")
        else:
            st.info("OpenAI generation disabled or no API key provided — showing retrieved contexts only.")
