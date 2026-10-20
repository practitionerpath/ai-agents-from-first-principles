import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import InvalidUpdateError


# ---------- State ----------

def merge_dicts(old: dict, new: dict) -> dict:
    """Reducer: new keys overwrite old ones, existing keys not in `new` are kept."""
    return {**old, **new}


class ChatState(TypedDict):
    messages: Annotated[list[str], operator.add]
    identifiers: Annotated[dict, merge_dicts]


class NoReducerState(TypedDict):
    """Same as ChatState, but `messages` has no reducer, so writes replace it."""
    messages: list[str]
    identifiers: Annotated[dict, merge_dicts]


class StatusState(TypedDict):
    """Single field, no reducer. Used to trigger a concurrent-write conflict."""
    status: str


# ---------- Nodes ----------

ORDER_ID_PATTERN = re.compile(r"ORD-\d{4}")


def extract_identifiers(state: dict) -> dict:
    """Run the regex on the latest message only, never the joined history."""
    latest = state["messages"][-1]
    match = ORDER_ID_PATTERN.search(latest)
    if match:
        return {"identifiers": {"order_id": match.group()}}
    return {}


def writer_a(state: StatusState) -> dict:
    return {"status": "written by writer_a"}


def writer_b(state: StatusState) -> dict:
    return {"status": "written by writer_b"}


# ---------- Raw build comparison ----------

TURNS = [
    "Where is my order ORD-1001?",
    "Sorry, wrong one, it's ORD-1007.",
]


def raw_build_extraction(turns: list[str]) -> str | None:
    """The raw build approach: regex over the joined conversation string."""
    joined = " ".join(turns)
    match = ORDER_ID_PATTERN.search(joined)
    return match.group() if match else None


# ---------- Graphs ----------

def build_extraction_graph(schema):
    """One-node graph. The schema decides which reducers apply."""
    builder = StateGraph(schema)
    builder.add_node("extract_identifiers", extract_identifiers)
    builder.add_edge(START, "extract_identifiers")
    builder.add_edge("extract_identifiers", END)
    return builder.compile(checkpointer=InMemorySaver())


def build_parallel_write_graph():
    """Two nodes run in the same superstep and both write `status`."""
    builder = StateGraph(StatusState)
    builder.add_node("writer_a", writer_a)
    builder.add_node("writer_b", writer_b)
    builder.add_edge(START, "writer_a")
    builder.add_edge(START, "writer_b")
    builder.add_edge("writer_a", END)
    builder.add_edge("writer_b", END)
    return builder.compile()


# ---------- Demos ----------

def run_turns(graph, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = {}
    for turn in TURNS:
        result = graph.invoke({"messages": [turn]}, config)
    return result


def demo_part_a():
    print("=== Part A: corrected identifier ===")
    result = run_turns(build_extraction_graph(ChatState), "part-a")
    print(f"Raw build (joined string): {raw_build_extraction(TURNS)}")
    print(f"LangGraph (per-turn, last write wins): {result['identifiers'].get('order_id')}")
    print(f"Messages in state: {result['messages']}")


def demo_part_b():
    print("\n=== Part B: concurrent writes, no reducer ===")
    graph = build_parallel_write_graph()
    try:
        graph.invoke({"status": "initial"})
        print("No error raised")
    except InvalidUpdateError as e:
        print(f"{type(e).__name__}: {str(e).splitlines()[0]}")


def demo_part_c():
    print("\n=== Part C: sequential writes, no reducer on messages ===")
    result = run_turns(build_extraction_graph(NoReducerState), "part-c")
    print(f"Identifier in state: {result['identifiers'].get('order_id')}")
    print(f"Messages in state: {result['messages']}")


if __name__ == "__main__":
    demo_part_a()
    demo_part_b()
    demo_part_c()