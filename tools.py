from langchain_core.tools import tool
import requests

# ---------------------------------------------------------
# WEB SEARCH TOOL (generic knowledge)
# ---------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """
    ALWAYS use this tool when answering factual or explanatory questions.
    Do not answer from memory.
    """
    # """
    # Search the web for a query and return relevant information.
    # """
    # Mocked search results
    return (
        f"Search results for '{query}':\n"
        # "- LangChain provides foundational LLM abstractions.\n"
        # "- LangGraph is built on top of LangChain.\n"
        # "- LangGraph enables stateful agent workflows."
    )

# ---------------------------------------------------------
# SUMMARIZER TOOL (final answer formatting)
# ---------------------------------------------------------

@tool
def summarize(text: str) -> str:
    """
    Summarize the given text if word count is more than 50.
    """
    return f"Summary: {text[:20]}..."



# ---------------------------------------------------------
# RAG PROBE TOOL (cheap, no LLM, no rerank)
# ---------------------------------------------------------
@tool
def rag_probe(query: str) -> dict:
    """
    Cheap probe to check whether private documents are likely
    to contain the answer.

    Calls /probe endpoint of rag-app.
    Returns:
      {
        "confidence": float,
        "hits": int,
        "latency": float
      }
    """
    response = requests.post(
        "http://localhost:8000/probe",
        json={"query": query},
        timeout=5
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------
# FULL RAG TOOL (expensive: retrieval + LLM)
# ---------------------------------------------------------
@tool
def rag_tool(query: str) -> dict:
    """
    Full RAG query against private documents.

    Calls /chat endpoint of rag-app.
    Returns:
      {
        "answer": str,
        "confidence": float,
        "hits": int
      }
    """
    response = requests.post(
        "http://localhost:8000/chat",
        json={"question": query},
        timeout=30
    )
    response.raise_for_status()
    return response.json()
