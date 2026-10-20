import os
import tiktoken
import chromadb
import trafilatura
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# The five public postmortem documents we are indexing.
# Each entry carries the metadata that will travel with every chunk.
DOCUMENTS = [
    {
        "url": "https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/",
        "company": "Cloudflare",
        "incident_date": "2019-07-02",
        "incident_title": "WAF Regex Backtracking Outage"
    },
    {
        "url": "https://blog.cloudflare.com/18-november-2025-outage/",
        "company": "Cloudflare",
        "incident_date": "2025-11-18",
        "incident_title": "Bot Management Feature File Outage"
    },
    {
        "url": "https://github.blog/news-insights/company-news/oct21-post-incident-analysis/",
        "company": "GitHub",
        "incident_date": "2018-10-21",
        "incident_title": "MySQL Failover Cascade"
    },
    {
        "url": "https://blog.cloudflare.com/cloudflare-service-outage-june-12-2025/",
        "company": "Cloudflare",
        "incident_date": "2025-06-12",
        "incident_title": "Workers KV Third-Party Dependency Failure"
    },
    {
        "url": "https://blog.cloudflare.com/cloudflare-outage-on-june-21-2022/",
        "company": "Cloudflare",
        "incident_date": "2022-06-21",
        "incident_title": "BGP Configuration Error"
    }
]

CHUNK_SIZE = 400        # tokens per chunk
CHUNK_OVERLAP = 80      # overlap between consecutive chunks
EMBEDDING_MODEL = "text-embedding-ada-002"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "postmortems"


def fetch_and_extract(url: str) -> str | None:
    """
    Fetch a URL and return clean article text.

    trafilatura identifies the main article body and strips everything
    else: navigation menus, footers, related-post links, cookie banners.
    Without this step, those fragments end up in the vector store and
    pollute retrieval results with irrelevant text.
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        print(f"  Could not fetch: {url}")
        return None

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True   # postmortems use tables for service impact summaries
    )

    if not text:
        print(f"  Could not extract article text from: {url}")
        return None

    return text


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping token-based chunks.

    We use tiktoken to count tokens because the embedding model
    measures its input limit in tokens, not words or characters.
    The overlap ensures that a sentence split across a chunk boundary
    still appears complete in at least one chunk.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)

        if end == len(tokens):
            break

        # Slide forward by (chunk_size - overlap) to create the next chunk
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings using the OpenAI embeddings API.

    Sending all chunks in one API call is more efficient than one call
    per chunk. OpenAI supports up to 2048 inputs per request, well above
    the chunk counts we will generate from these five documents.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def build_collection(reset: bool = True):
    """
    Create (or recreate) the ChromaDB collection.

    cosine distance is the right choice for OpenAI embeddings because
    those vectors are already normalized. Cosine similarity measures
    the angle between vectors, not their magnitude, which is exactly
    what semantic similarity search needs.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"Cleared existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def ingest_document(doc: dict, collection) -> int:
    """
    Full ingestion pipeline for one document.
    Returns the number of chunks stored, or 0 on failure.
    """
    print(f"\n{doc['incident_title']} | {doc['company']} | {doc['incident_date']}")

    print("  Fetching and extracting...")
    text = fetch_and_extract(doc["url"])
    if not text:
        return 0

    print(f"  Extracted {len(text.split())} words")

    print("  Chunking...")
    chunks = chunk_text(text)
    print(f"  {len(chunks)} chunks at {CHUNK_SIZE} tokens with {CHUNK_OVERLAP} overlap")

    print("  Embedding...")
    embeddings = embed_texts(chunks)

    print("  Storing in ChromaDB...")

    # IDs must be unique across all documents in the collection.
    # Using company + date + chunk index guarantees uniqueness.
    id_prefix = f"{doc['company'].lower().replace(' ', '_')}_{doc['incident_date']}"
    ids = [f"{id_prefix}_chunk_{i}" for i in range(len(chunks))]

    # Metadata travels with every chunk so the retriever can return
    # source attribution alongside the text of each result.
    metadatas = [
        {
            "source_url": doc["url"],
            "company": doc["company"],
            "incident_date": doc["incident_date"],
            "incident_title": doc["incident_title"],
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    print(f"  Done. {len(chunks)} chunks stored.")
    return len(chunks)


def main():
    print("=== Postmortem Analyzer: Ingestion ===\n")

    collection = build_collection(reset=True)

    total_chunks = 0
    successful = 0

    for doc in DOCUMENTS:
        try:
            count = ingest_document(doc, collection)
            if count > 0:
                total_chunks += count
                successful += 1
        except Exception as e:
            print(f"  Error on {doc['incident_title']}: {e}")

    print(f"\n=== Complete ===")
    print(f"Documents indexed: {successful} / {len(DOCUMENTS)}")
    print(f"Total chunks stored: {total_chunks}")
    print(f"Collection '{COLLECTION_NAME}' saved to {CHROMA_PATH}/")


if __name__ == "__main__":
    main()