"""
graph.py
========
Wire tất cả nodes và conditional edges thành một CompiledStateGraph.

Dùng:
    from graph import build_graph
    graph = build_graph()
    result = graph.invoke({"current_question": "...", "messages": []})
    print(result["final_answer"])
"""

from langgraph.graph import END, START, StateGraph

from graph_nodes import (
    check_recency,
    generate_answer,
    process_input,
    retrieve_initial,
    route_after_recency_check,
)
from graph_state import GraphState


def build_graph():
    builder = StateGraph(GraphState)

    # ── Đăng ký nodes ─────────────────────────────────────────────────────────
    builder.add_node("process_input",        process_input)
    builder.add_node("retrieve_initial",     retrieve_initial)
    builder.add_node("check_recency",        check_recency)
    builder.add_node("generate_answer",      generate_answer)

    # ── Edges tuyến tính ──────────────────────────────────────────────────────
    builder.add_edge(START,                "process_input")
    builder.add_edge("process_input",      "retrieve_initial")
    builder.add_edge("retrieve_initial",   "check_recency")

    # ── Conditional: sau check_recency ────────────────────────────────────────
    # RECENT / OLD     → generate_answer
    # NOT_FOUND + full → generate_answer
    builder.add_conditional_edges(
        "check_recency",
        route_after_recency_check,
        {
            "generate_answer": "generate_answer",
        },
    )

    # ── End ───────────────────────────────────────────────────────────────────
    builder.add_edge("generate_answer", END)

    return builder.compile()


# Singleton — import 1 lần, dùng lại
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph