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
    crawl_and_process,
    decide_crawl,
    generate_answer,
    process_input,
    retrieve_after_crawl,
    retrieve_initial,
    route_after_decide_crawl,
    route_after_recency_check,
)
from graph_state import GraphState


def build_graph():
    builder = StateGraph(GraphState)

    # ── Đăng ký nodes ─────────────────────────────────────────────────────────
    builder.add_node("process_input",        process_input)
    builder.add_node("retrieve_initial",     retrieve_initial)
    builder.add_node("check_recency",        check_recency)
    builder.add_node("decide_crawl",         decide_crawl)
    builder.add_node("crawl_and_process",    crawl_and_process)
    builder.add_node("retrieve_after_crawl", retrieve_after_crawl)
    builder.add_node("generate_answer",      generate_answer)

    # ── Edges tuyến tính ──────────────────────────────────────────────────────
    builder.add_edge(START,                "process_input")
    builder.add_edge("process_input",      "retrieve_initial")
    builder.add_edge("retrieve_initial",   "check_recency")

    # ── Conditional: sau check_recency ────────────────────────────────────────
    # RECENT / OLD     → generate_answer
    # NOT_FOUND + slot → decide_crawl
    # NOT_FOUND + full → generate_answer
    builder.add_conditional_edges(
        "check_recency",
        route_after_recency_check,
        {
            "generate_answer": "generate_answer",
            "decide_crawl":    "decide_crawl",
        },
    )

    # ── Conditional: sau decide_crawl ─────────────────────────────────────────
    # Có URL   → crawl_and_process
    # Không URL → generate_answer
    builder.add_conditional_edges(
        "decide_crawl",
        route_after_decide_crawl,
        {
            "crawl_and_process": "crawl_and_process",
            "generate_answer":   "generate_answer",
        },
    )

    # ── Sau crawl: retrieve lại → check_recency lại (loop tối đa 2 lần) ──────
    builder.add_edge("crawl_and_process",    "retrieve_after_crawl")
    builder.add_edge("retrieve_after_crawl", "check_recency")

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