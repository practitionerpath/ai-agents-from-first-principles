import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./chroma_store")


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def index_documents(documents_path="./documents"):
    try:
        chroma_client.delete_collection("tickets")
        print("Cleared existing collection.")
    except Exception:
        pass

    collection = chroma_client.create_collection("tickets")

    for filename in sorted(os.listdir(documents_path)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(documents_path, filename)
        with open(filepath, "r") as f:
            text = f.read()

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            embedding = embed_text(chunk)
            chunk_id = f"{filename}_{i}"

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": filename, "chunk_index": i}],
                ids=[chunk_id]
            )
            print(f"Indexed: {chunk_id}")

    print(f"\nIndexing complete. Total chunks in store: {collection.count()}")


if __name__ == "__main__":
    index_documents()