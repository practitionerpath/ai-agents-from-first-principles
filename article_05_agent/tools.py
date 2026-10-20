from retriever import search_chroma


def retrieve_context(query):
    chunks = search_chroma(query, top_k=2)

    if not chunks:
        return "No relevant information found for this query."

    result = ""
    for chunk in chunks:
        result += f"\n[Source: {chunk['source']}, chunk {chunk['chunk_index']}]\n{chunk['text']}\n"

    return result.strip()


TOOL_FUNCTIONS = {
    "retrieve_context": retrieve_context
}