# SemaGC: Semantic Garbage Collection Runtime

A context management system for multi-agent LLM pipelines using LangGraph.

## What it does
SemaGC reduces context state size by 42.9% in multi-agent pipelines by:
- Pruning semantically drifted messages using cosine similarity
- Compressing relevant intermediate messages into a Micro-Rationale
- Operating transparently as a LangGraph state reducer

## Results
| Task | Baseline | SemaGC | Reduction |
|---|---|---|---|
| FastAPI REST API | 7 msgs | 4 msgs | 42.9% |
| Data Pipeline | 7 msgs | 4 msgs | 42.9% |
| CLI Monitor Tool | 7 msgs | 4 msgs | 42.9% |
| Web Scraper | 7 msgs | 4 msgs | 42.9% |
| Caching System | 7 msgs | 4 msgs | 42.9% |
| **Average** | **7** | **4** | **42.9%** |

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/semagc-runtime
cd semagc-runtime
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add your API key to `.env`:
GROQ_API_KEY=your_key_here

Run the dashboard:
```bash
streamlit run app.py
```

## Architecture
- `semagc_runtime.py` — Core GC engine and LangGraph reducer
- `graph_pipeline.py` — 6-node SemaGC pipeline
- `baseline_pipeline.py` — Control pipeline (no GC)
- `app.py` — Streamlit comparison dashboard
- `benchmarks.py` — Automated benchmark runner

## Paper
[Link to arXiv paper — add after submission]