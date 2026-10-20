import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-ada-002"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "postmortems"

# Load the collection once at module import time.
# Both the agent loop and the verifier re-retrieval pass import this
# module, so we only want one connection to ChromaDB across both callers.
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma_client.get_collection(name=COLLECTION_NAME)


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed a query and return the top_k most similar chunks from ChromaDB.

    Each result is a dict with three keys:
      text      — the raw chunk text
      metadata  — company, incident_date, incident_title, source_url, chunk_index
      distance  — cosine distance (lower = more similar)

    The caller is responsible for formatting these results into
    context for the LLM. Keeping formatting out of this function
    means the agent and verifier can each format results differently
    without changing retrieval logic.
    """
    # Embed the query using the same model used at ingestion time.
    # Using a different model here would put the query in a different
    # vector space and similarity scores would be meaningless.
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query]
    )
    query_embedding = response.data[0].embedding

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # ChromaDB returns parallel lists. Zip them into a list of dicts
    # so callers can treat each result as a self-contained object.
    chunks = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": text,
            "metadata": metadata,
            "distance": distance
        })

    return chunks


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a labeled context block for the LLM.

    Each chunk is prefixed with its source so the LLM can attribute
    claims to specific documents. Without this labeling, the LLM has
    no way to distinguish which fact came from which incident.
    """
    if not chunks:
        return "No relevant content found in the indexed documents."

    sections = []
    for chunk in chunks:
        m = chunk["metadata"]
        header = f"[{m['company']} | {m['incident_date']} | {m['incident_title']}]"
        sections.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(sections)


if __name__ == "__main__":
    # Quick smoke test — run this file directly to verify retrieval works
    # before wiring it into the agent.
    test_query = "what caused the outage"
    print(f"Test query: '{test_query}'\n")

    results = search(test_query, top_k=3)

    for i, chunk in enumerate(results, 1):
        m = chunk["metadata"]
        print(f"Result {i}")
        print(f"  Source:   {m['company']} | {m['incident_date']} | {m['incident_title']}")
        print(f"  Distance: {chunk['distance']:.4f}")
        print(f"  Preview:  {chunk['text'][:120].strip()}...")
        print()