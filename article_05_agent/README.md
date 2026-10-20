# article_05_agent

## Activate
```bash
source ../.venv/bin/activate
```

## Run Order
```bash
python3 indexer.py
```
Chunks all documents, embeds them, and stores in ChromaDB. Run once before anything else, and re-run if any document changes.

```bash
python3 basic_rag.py
```
Runs two queries through the fixed RAG pipeline to show where single-hop retrieval works and where multi-hop retrieval fails.

```bash
python3 agent.py
```
Runs the same two queries through the agentic loop to show multi-hop retrieval succeeding where basic RAG could not.
