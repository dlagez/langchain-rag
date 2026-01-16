# LangChain RAG Starter

Minimal RAG project skeleton using LangChain + Qdrant + Google Gemini embeddings.

## Quickstart (PowerShell)
1. Activate venv: `.\.venv\Scripts\Activate.ps1`
2. Install deps: `python -m pip install -r requirements.txt`
3. Create `.env` with your key: `GOOGLE_API_KEY=...`
4. (Optional) Set `GOOGLE_MODEL` to a supported model, e.g. `gemini-2.5-flash`
4. Add `.txt` files to `data/`
5. Run: `python src/rag_demo.py`

Notes:
- The vector store persists to `qdrant/`. Delete that folder to rebuild.
