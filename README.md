# Simple RAG LLM (minimal)
This RAG-LLM uses standard partitioning tehcniques to obtain 
to extract the useful elements which can be the processed by the LLM 
to produce a result that is based upon the RAG mechanism

Steps to follow to run the LLM:
1. Create a virtual python environment
> For mac Users
`python -m venv <venv_name>`
`source <venv_name>/bin/activate`
> For Windows users
`python -m venv <venv_name>`
`.<venv_name>\Scripts\activate`
2. After you are done setting the environment, run this command in your terminal
below to access the UI using streamlit
`streamlit run app.py`

WARNING: Do not index sensitive/proprietary data without permission.
