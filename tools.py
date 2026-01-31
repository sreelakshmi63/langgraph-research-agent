# from langchain_core.tools import tool

# @tool
# def web_search(query: str) -> str:
#     print(f"\n🛠 web_search CALLED with query: {query}")
#     return (
#         f"Search results for '{query}':\n"
#         "- LangGraph is a library for building stateful LLM agents.\n"
#         "- It is built on top of LangChain.\n"
#         "- It provides explicit control over agent workflows."
#     )

# @tool
# def summarize(text: str) -> str:
#     print("\n🛠 summarize CALLED")
#     return f"Summary: {text[:200]}..."




from langchain_core.tools import tool

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


@tool
def summarize(text: str) -> str:
    """
    Summarize the given text if word count is more than 50.
    """
    return f"Summary: {text[:20]}..."
