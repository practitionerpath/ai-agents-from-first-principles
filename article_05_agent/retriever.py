from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./chroma_store")


def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding


def search_chroma(query, top_k=2):
    collection = chroma_client.get_collection("tickets")
    query_vector = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"]
        })

    return chunks