from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from tools import web_search, summarize, rag_probe, rag_tool


from dotenv import load_dotenv
load_dotenv()
# -----------------------------
# 1. Define State
# -----------------------------
# class AgentState(TypedDict):
#     messages: Annotated[list[BaseMessage], "conversation history"]

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int


# -----------------------------
# 2. LLM + tools
# -----------------------------
llm = ChatOpenAI(
    # model="gpt-4o-mini",
    model="gpt-3.5-turbo",
    temperature=0
)

tools = [web_search, summarize, rag_probe, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -----------------------------
# 3. Agent node
# -----------------------------

# SYSTEM_PROMPT = SystemMessage(
#     content=(
#         "You are a research assistant.\n"
#         "Use tools when needed to gather information.\n"
#         "Once you have enough information, provide a FINAL answer.\n"
#         "Do NOT call any tools after providing the final answer."
#     )
# )



# SYSTEM_PROMPT = """
# You are a research assistant.

# You may use tools to gather information.
# Once you have enough information, STOP calling tools
# and produce a FINAL ANSWER in plain text.

# Rules:
# - If you do not need a tool, answer directly.
# - Do NOT call tools endlessly.
# - When answering, DO NOT call any tools.
# """


# SYSTEM_PROMPT = """
# You are a research assistant.

# You may use tools to gather information.
# Once you have enough information, STOP calling tools
# and produce a FINAL ANSWER in plain text.

# When multiple web search results are available:
# 1. Call the summarize tool to condense them
# 2. Then produce a final answer

# Rules:
# - If you do not need a tool, answer directly.
# - Do NOT call tools endlessly.
# - When answering, DO NOT call any tools.
# """

SYSTEM_PROMPT = """
You are a research assistant with access to tools.

Routing rules (STRICT, DO NOT VIOLATE):
1. ALWAYS call `rag_probe` first for every user question only once
2. Read the returned `confidence` value.
3. If confidence >= {CONFIDENCE_THRESHOLD}, call `rag_tool`.
4. If confidence < {CONFIDENCE_THRESHOLD}, DO NOT call `rag_tool`;
   instead call `web_search`.
5. After gathering information, call the summarize tool to condense them
6. Then produce a final answer

Rules:
- Do not hallucinate document content.
- Do not skip the probe step.
- Do not call tools endlessly.
- Produce only the final summarized answer.
"""






def agent_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    # messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    # return {"messages": [response], "steps": state["steps"] + 1}
    return {
        "messages": state["messages"] + [response],
        "steps": state["steps"] + 1
    }
# -----------------------------
# 4. Tool node
# -----------------------------
tool_node = ToolNode(tools)

# -----------------------------
# 5. Build graph
# -----------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

# Conditional routing
# graph.add_conditional_edges(
#     "agent",
#     lambda state: (
#         "tools"
#         if isinstance(state["messages"][-1], AIMessage)
#         and state["messages"][-1].tool_calls
#         else END
#     )
# )

def route(state: AgentState):
    last = state["messages"][-1]

    # Hard stop after 10 agent turns
    if state["steps"] >= 15:
        print("Force termination initiated after 15 steps of agent")
        return END

    # Use tools only if explicitly requested
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"

    print("Termiinating agent as no tool calls detected.")
    return END


graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route)
app = graph.compile()
