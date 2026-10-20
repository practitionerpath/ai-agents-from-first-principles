"""Wires the ten nodes into a graph and compiles it.

Two conditional edges, each with an explicit list of allowed destinations. The
list lets compile() check that every destination exists, and makes every branch
visible in the rendered diagram. Every other edge is fixed.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

import nodes
import specialists
from routing import route_after_specialist, route_after_supervisor
from state import ChatState

SPECIALISTS = {
    "order_management": specialists.order_management,
    "returns_refunds": specialists.returns_refunds,
    "billing_payments": specialists.billing_payments,
}

AFTER_SUPERVISOR = [*SPECIALISTS, "clarify", "escalation", "assemble_payload"]


def after_specialist(name: str) -> list[str]:
    """A specialist can hand off to the other specialists, never to itself.

    It stamps its result for the current order, so it is always fresh afterwards.
    Leaving itself off the list keeps the diagram honest, and turns a routing bug
    that tried it into a loud error instead of a loop.
    """
    return [s for s in SPECIALISTS if s != name] + ["escalation", "assemble_payload"]


def build_graph(checkpointer=None):
    graph = StateGraph(ChatState)

    graph.add_node("extract_identifiers", nodes.extract_identifiers)
    graph.add_node("resolve_order", nodes.resolve_order)
    graph.add_node("supervisor", nodes.supervisor)
    for name, fn in SPECIALISTS.items():
        graph.add_node(name, fn)
    graph.add_node("clarify", nodes.clarify)
    graph.add_node("escalation", nodes.escalation)
    graph.add_node("assemble_payload", nodes.assemble_payload)
    graph.add_node("write_response", nodes.write_response)

    # Front of the graph: fixed. Every turn, including a resumed clarification,
    # enters through the same two code-only nodes.
    graph.add_edge(START, "extract_identifiers")
    graph.add_edge("extract_identifiers", "resolve_order")
    graph.add_edge("resolve_order", "supervisor")

    # The two decision points.
    graph.add_conditional_edges("supervisor", route_after_supervisor, AFTER_SUPERVISOR)
    for name in SPECIALISTS:
        graph.add_conditional_edges(name, route_after_specialist, after_specialist(name))

    # The only cycle: a reply goes back through extraction and resolution.
    graph.add_edge("clarify", "extract_identifiers")

    # Back of the graph: fixed.
    graph.add_edge("escalation", "assemble_payload")
    graph.add_edge("assemble_payload", "write_response")
    graph.add_edge("write_response", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


if __name__ == "__main__":
    print(build_graph().get_graph().draw_mermaid())