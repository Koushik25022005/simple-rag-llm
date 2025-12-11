# Simple RAG LLM (minimal)

1. Copy `.env.example` → `.env` and set values.
2. Install: `pip install -r requirements.txt`
3. Put documents into `data/` (plain text or simple .txt).
4. Build index:
   `python src/build_index.py`
5. Query:
   `python src/query.py "Your question goes here"`
or run API:
   `uvicorn src.app:app --reload --port 8000`

WARNING: Do not index sensitive/proprietary data without permission.
