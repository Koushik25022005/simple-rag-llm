# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.query import retrieve, answer_with_openai
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class Q(BaseModel):
    question: str

@app.post("/query")
def query(q: Q):
    ctxs = retrieve(q.question)
    if OPENAI_KEY:
        ans = answer_with_openai(q.question, ctxs)
        return {"answer": ans, "contexts": ctxs}
    else:
        return {"answer": None, "contexts": ctxs, "note": "No OPENAI_API_KEY set; only contexts returned."}
