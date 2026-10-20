import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


MAX_CLARIFICATION_ATTEMPTS = 2
ALL_ROUTES = ["clarify", "order_management"]
ORDER_ID_PATTERN = re.compile(r"ORD-\d{4}")


# ---------- State ----------

def merge_dicts(old: dict, new: dict) -> dict:
    """Reducer: new keys overwrite old ones, existing keys not in `new` are kept."""
    return {**old, **new}


class RouterState(TypedDict):
    messages: Annotated[list[str], operator.add]
    identifiers: Annotated[dict, merge_dicts]
    route: str
    clarification_attempts: int


# ---------- Code decides: the guard ----------

def compute_allowed_routes(state: RouterState, identifier_guard: bool) -> list[str]:
    """Remove routes that state proves are wrong. The supervisor picks from what remains."""
    allowed = list(ALL_ROUTES)
    has_order_id = bool(state.get("identifiers", {}).get("order_id"))
    attempts = state.get("clarification_attempts", 0)

    if identifier_guard and has_order_id:
        allowed.remove("clarify")
    elif attempts >= MAX_CLARIFICATION_ATTEMPTS:
        allowed.remove("clarify")
    return allowed


# ---------- LLM reasons: the scripted stub ----------

def stub_supervisor_llm(allowed_routes: list[str]) -> str:
    """Reproduces the documented misjudgment: prefers clarify whenever it is allowed."""
    return "clarify" if "clarify" in allowed_routes else "order_management"


# ---------- Nodes ----------

def extract_identifiers(state: RouterState) -> dict:
    """Run the regex on the latest message only."""
    match = ORDER_ID_PATTERN.search(state["messages"][-1])
    if match:
        return {"identifiers": {"order_id": match.group()}}
    return {}


def make_supervisor(identifier_guard: bool):
    def supervisor(state: RouterState) -> dict:
        allowed = compute_allowed_routes(state, identifier_guard)
        route = stub_supervisor_llm(allowed)
        print(f"  supervisor: allowed={allowed} -> {route}")
        return {"route": route}
    return supervisor


def clarify(state: RouterState) -> dict:
    attempts = state.get("clarification_attempts", 0) + 1
    print(f"  clarify: asking customer for order ID (attempt {attempts})")
    return {"clarification_attempts": attempts}


def order_management(state: RouterState) -> dict:
    print(f"  order_management: looking up {state['identifiers']['order_id']}")
    return {}


def route_after_supervisor(state: RouterState) -> str:
    """Dumb mapping. All routing logic lives in the supervisor node."""
    return state["route"]


# ---------- Graph ----------

def build_graph(identifier_guard: bool):
    builder = StateGraph(RouterState)
    builder.add_node("extract_identifiers", extract_identifiers)
    builder.add_node("supervisor", make_supervisor(identifier_guard))
    builder.add_node("clarify", clarify)
    builder.add_node("order_management", order_management)

    builder.add_edge(START, "extract_identifiers")
    builder.add_edge("extract_identifiers", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"clarify": "clarify", "order_management": "order_management"},
    )
    builder.add_edge("clarify", END)
    builder.add_edge("order_management", END)
    return builder.compile(checkpointer=InMemorySaver())


# ---------- Demo ----------

TURNS = [
    "Where is my order?",
    "It's ORD-1011.",
]


def run(identifier_guard: bool, thread_id: str):
    label = "ON" if identifier_guard else "OFF"
    print(f"\n=== Identifier guard {label} ===")
    graph = build_graph(identifier_guard)
    config = {"configurable": {"thread_id": thread_id}}

    for i, turn in enumerate(TURNS, start=1):
        print(f"Turn {i}: customer says {turn!r}")
        graph.invoke({"messages": [turn]}, config)


if __name__ == "__main__":
    run(identifier_guard=False, thread_id="guard-off")
    run(identifier_guard=True, thread_id="guard-on")