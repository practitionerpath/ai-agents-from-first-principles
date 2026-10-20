from openai import OpenAI
from retriever import search_chroma
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def build_prompt(question, chunks):
    context = ""
    for chunk in chunks:
        context += f"\n[Source: {chunk['source']}, chunk {chunk['chunk_index']}]\n{chunk['text']}\n"

    return f"""Answer the question using only the context provided below. 
If the context does not contain enough information to answer fully, say so explicitly.

Context:
{context}

Question: {question}

Answer:"""


def run_basic_rag(question):
    print(f"\nQuery: {question}")
    print("-" * 50)

    chunks = search_chroma(question, top_k=2)

    print("Retrieved chunks:")
    for chunk in chunks:
        print(f"  - {chunk['source']} (chunk {chunk['chunk_index']})")

    prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    print(f"\nAnswer:\n{answer}")
    return answer


if __name__ == "__main__":
    print("=" * 60)
    print("QUERY 1: Single-hop (should work)")
    print("=" * 60)
    run_basic_rag("What symptoms were users reporting during the authentication incident?")

    print("\n" + "=" * 60)
    print("QUERY 2: Multi-hop (watch what gets retrieved)")
    print("=" * 60)
    run_basic_rag("What caused the authentication failures and how was it fixed?")