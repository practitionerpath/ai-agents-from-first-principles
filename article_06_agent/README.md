# article_06_agent

## Activate
```bash
source ../.venv/bin/activate
```

## Run Order
```bash
python3 ingest.py
```
Fetches five public postmortem URLs, extracts article text, chunks and embeds each document, and stores 59 chunks in a local ChromaDB collection. Run once before starting the agent.

```bash
python3 retriever.py
```
Embeds a test query and prints the top three retrieved chunks with source labels and similarity distances. Run this to confirm the index is healthy before starting the agent.

```bash
python3 agent.py
```
Starts the conversational loop: retrieves from ChromaDB, generates answers, verifies each factual claim against retrieved chunks, and flags anything that could not be grounded.

## Test Queries

Run these in order. Queries 3 and 4 depend on the previous answer being in conversation history.

```
What caused the Cloudflare July 2019 outage?
```
```
Which of the indexed incidents were caused by a configuration or deployment change?
```
```
How did their incident response process compare?
```
```
What caused the AWS us-east-1 outage in 2021?
```
