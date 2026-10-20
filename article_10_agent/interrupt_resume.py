import operator
import os
import re
import sqlite3
import sys
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt


DB_PATH = "checkpoints.db"
THREAD_ID = "customer-conversation"
MAX_CLARIFICATION_ATTEMPTS = 2
ALL_ROUTES = ["clarify", "order_management"]
ORDER_ID_PATTERN = re.compile(r"ORD-\d{4}")
FIRST_TURN = "Where is my order?"


# ---------- State ----------

def merge_dicts(old: dict, new: dict) -> dict:
    return {**old, **new}


class ConversationState(TypedDict):
    messages: Annotated[list[str], operator.add]
    identifiers: Annotated[dict, merge_dicts]
    route: str
    clarification_attempts: int


# ---------- Guard and stub (same as routing_guard.py, guard always on) ----------

def compute_allowed_routes(state: ConversationState) -> list[str]:
    allowed = list(ALL_ROUTES)
    has_order_id = bool(state.get("identifiers", {}).get("order_id"))
    attempts = state.get("clarification_attempts", 0)
    if has_order_id or attempts >= MAX_CLARIFICATION_ATTEMPTS:
        allowed.remove("clarify")
    return allowed


def stub_supervisor_llm(allowed_routes: list[str]) -> str:
    return "clarify" if "clarify" in allowed_routes else "order_management"


# ---------- Nodes ----------

def extract_identifiers(state: ConversationState) -> dict:
    match = ORDER_ID_PATTERN.search(state["messages"][-1])
    if match:
        return {"identifiers": {"order_id": match.group()}}
    return {}


def supervisor(state: ConversationState) -> dict:
    allowed = compute_allowed_routes(state)
    route = stub_supervisor_llm(allowed)
    print(f"  supervisor: allowed={allowed} -> {route}")
    return {"route": route}


def clarify(state: ConversationState) -> dict:
    attempts = state.get("clarification_attempts", 0) + 1
    # Deliberately placed BEFORE interrupt(): this line runs again on resume.
    print(f"  clarify: side effect, sending clarification message (attempt {attempts})")
    reply = interrupt({"question": "Could you share your order ID?"})
    print(f"  clarify: received reply {reply!r}")
    return {"messages": [reply], "clarification_attempts": attempts}


def order_management(state: ConversationState) -> dict:
    print(f"  order_management: looking up {state['identifiers']['order_id']}")
    return {}


def route_after_supervisor(state: ConversationState) -> str:
    return state["route"]


# ---------- Graph ----------

def build_graph(checkpointer):
    builder = StateGraph(ConversationState)
    builder.add_node("extract_identifiers", extract_identifiers)
    builder.add_node("supervisor", supervisor)
    builder.add_node("clarify", clarify)
    builder.add_node("order_management", order_management)

    builder.add_edge(START, "extract_identifiers")
    builder.add_edge("extract_identifiers", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"clarify": "clarify", "order_management": "order_management"},
    )
    builder.add_edge("clarify", "extract_identifiers")  # single ingress: replies re-enter here
    builder.add_edge("order_management", END)
    return builder.compile(checkpointer=checkpointer)


# ---------- Commands ----------

def cmd_start(graph, config):
    print(f"Customer says {FIRST_TURN!r}")
    result = graph.invoke({"messages": [FIRST_TURN]}, config)
    if "__interrupt__" in result:
        question = result["__interrupt__"][0].value["question"]
        print(f"Paused at: {graph.get_state(config).next}")
        print(f"Waiting for customer reply to: {question!r}")


def cmd_resume(graph, config, reply: str):
    print(f"Resuming with customer reply {reply!r}")
    graph.invoke(Command(resume=reply), config)
    print(f"Paused at: {graph.get_state(config).next or 'nothing, graph finished'}")


def cmd_history(graph, config):
    snapshots = list(graph.get_state_history(config))
    for snap in reversed(snapshots):  # history is returned newest first
        step = snap.metadata.get("step")
        ids = snap.values.get("identifiers", {})
        attempts = snap.values.get("clarification_attempts", 0)
        print(f"step {step:>2} | next={snap.next} | identifiers={ids} | attempts={attempts}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"start", "resume", "history"}:
        print('Usage: interrupt_resume.py start | resume "<reply>" | history')
        sys.exit(1)

    command = sys.argv[1]
    if command == "start" and os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # clean slate for each demo run

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    graph = build_graph(SqliteSaver(conn))
    config = {"configurable": {"thread_id": THREAD_ID}}

    if command == "start":
        cmd_start(graph, config)
    elif command == "resume":
        if len(sys.argv) < 3:
            print('Usage: interrupt_resume.py resume "<reply>"')
            sys.exit(1)
        cmd_resume(graph, config, sys.argv[2])
    else:
        cmd_history(graph, config)

    conn.close()


if __name__ == "__main__":
    main()