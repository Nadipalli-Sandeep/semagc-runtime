import time
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import Annotated, TypedDict, List, Optional
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# ── Plain reducer — just appends, no GC ───────────────────────────────────────
def plain_reducer(existing: List[BaseMessage], new: List[BaseMessage]) -> List[BaseMessage]:
    return existing + new

# ── Baseline State — identical structure but NO GC reducer ────────────────────
class BaselineState(TypedDict):
    memory_vault:      Annotated[List[BaseMessage], plain_reducer]
    execution_metrics: dict
    current_task:      str
    task_result:       Optional[str]

# ── Node 1: Task Planner ──────────────────────────────────────────────────────
def planner_node(state: BaselineState) -> dict:
    time.sleep(4)
    task = state["current_task"]
    response = llm.invoke([
        HumanMessage(content=f"List 3 subtasks for: {task}")
    ])
    return {
        "memory_vault": [AIMessage(content=f"[Planner]: {response.content}")],
        "execution_metrics": {**state.get("execution_metrics", {}), "planner": "done"}
    }

# ── Node 2: Code Generator ────────────────────────────────────────────────────
def code_gen_node(state: BaselineState) -> dict:
    time.sleep(4)
    history = "\n".join([m.content for m in state["memory_vault"]])[-2000:]
    response = llm.invoke([
        HumanMessage(content=f"Write Python code for this plan:\n{history}")
    ])
    return {
        "memory_vault": [AIMessage(content=f"[CodeGen]: {response.content}")],
        "execution_metrics": {**state.get("execution_metrics", {}), "codegen": "done"}
    }

# ── Node 3: Debugger ──────────────────────────────────────────────────────────
def debugger_node(state: BaselineState) -> dict:
    time.sleep(4)
    history = "\n".join([m.content for m in state["memory_vault"]])[-2000:]
    response = llm.invoke([
        HumanMessage(content=f"Fix bugs in this code:\n{history}")
    ])
    return {
        "memory_vault": [AIMessage(content=f"[Debugger]: {response.content}")],
        "execution_metrics": {**state.get("execution_metrics", {}), "debugger": "done"}
    }

# ── Node 4: Code Reviewer ─────────────────────────────────────────────────────
def reviewer_node(state: BaselineState) -> dict:
    time.sleep(4)
    history = "\n".join([m.content for m in state["memory_vault"]])[-2000:]
    response = llm.invoke([
        HumanMessage(content=f"Review for security and best practices:\n{history}")
    ])
    return {
        "memory_vault": [AIMessage(content=f"[Reviewer]: {response.content}")],
        "execution_metrics": {**state.get("execution_metrics", {}), "reviewer": "done"}
    }

# ── Node 5: Test Writer ───────────────────────────────────────────────────────
def test_writer_node(state: BaselineState) -> dict:
    time.sleep(4)
    history = "\n".join([m.content for m in state["memory_vault"]])[-2000:]
    response = llm.invoke([
        HumanMessage(content=f"Write unit tests for this code:\n{history}")
    ])
    return {
        "memory_vault": [AIMessage(content=f"[TestWriter]: {response.content}")],
        "execution_metrics": {**state.get("execution_metrics", {}), "test_writer": "done"}
    }

# ── Node 6: Final Summarizer ──────────────────────────────────────────────────
def summarizer_node(state: BaselineState) -> dict:
    time.sleep(4)
    history = "\n".join([m.content for m in state["memory_vault"]])[-2000:]
    response = llm.invoke([
        HumanMessage(content=f"Summarize the final working code and explain it briefly:\n{history}")
    ])
    return {
        "memory_vault": [AIMessage(content=f"[Summary]: {response.content}")],
        "task_result": response.content,
        "execution_metrics": {**state.get("execution_metrics", {}), "done": True}
    }

# ── Build baseline graph ──────────────────────────────────────────────────────
def build_baseline_graph():
    builder = StateGraph(BaselineState)

    builder.add_node("planner",     planner_node)
    builder.add_node("code_gen",    code_gen_node)
    builder.add_node("debugger",    debugger_node)
    builder.add_node("reviewer",    reviewer_node)
    builder.add_node("test_writer", test_writer_node)
    builder.add_node("summarizer",  summarizer_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner",     "code_gen")
    builder.add_edge("code_gen",    "debugger")
    builder.add_edge("debugger",    "reviewer")
    builder.add_edge("reviewer",    "test_writer")
    builder.add_edge("test_writer", "summarizer")
    builder.add_edge("summarizer",  END)

    return builder.compile(checkpointer=MemorySaver())

# ── Run it ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    graph = build_baseline_graph()

    initial_state = {
        "memory_vault": [HumanMessage(content="Build a complete Python REST API using FastAPI that includes user authentication with JWT tokens, a PostgreSQL database connection, CRUD operations for a todo list, input validation, error handling, rate limiting, and full test coverage")],
        "execution_metrics": {},
        "current_task": "Build a complete Python REST API using FastAPI that includes user authentication with JWT tokens, a PostgreSQL database connection, CRUD operations for a todo list, input validation, error handling, rate limiting, and full test coverage",
        "task_result": None,
    }

    config = {"configurable": {"thread_id": "baseline-run-1"}}
    result = graph.invoke(initial_state, config)

    print("\n" + "="*50)
    print("BASELINE RESULT")
    print("="*50)
    print(result["task_result"])
    print("\nFinal vault size:", len(result["memory_vault"]), "messages")
    print("Execution metrics:", result["execution_metrics"])