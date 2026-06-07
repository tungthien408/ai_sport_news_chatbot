"""
graph.py
========
Wire all nodes and conditional edges into a CompiledStateGraph.

Uses GraphNodeManager for OOP-based node registration.

Usage:
    from graph import get_graph
    graph = get_graph()
    result = graph.invoke({"current_question": "...", "messages": []})
    print(result["final_answer"])
"""

from langgraph.graph import END, START, StateGraph

from graph_nodes import GraphNodeManager
from graph_state import GraphState


def build_graph():
    """Build and compile the LangGraph workflow.

    Flow:
        START → process_input → classify_question
            ├─ sports_news  → retrieve_initial → check_recency
            │                                     ├─ RECENT/OLD  → generate_answer → END
            │                                     └─ NOT_FOUND   → wikipedia_search → generate_answer → END
            ├─ sports_wiki  → wikipedia_search → generate_answer → END
            └─ off_topic    → generate_answer (polite decline) → END
    """
    manager = GraphNodeManager()
    builder = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("process_input",     manager.process_input)
    builder.add_node("classify_question", manager.classify_question)
    builder.add_node("retrieve_initial",  manager.retrieve_initial)
    builder.add_node("check_recency",     manager.check_recency)
    builder.add_node("wikipedia_search",  manager.wikipedia_search)
    builder.add_node("generate_answer",   manager.generate_answer)

    # ── Linear edges ─────────────────────────────────────────────────────────
    builder.add_edge(START,              "process_input")
    builder.add_edge("process_input",    "classify_question")
    builder.add_edge("retrieve_initial", "check_recency")
    builder.add_edge("wikipedia_search", "generate_answer")
    builder.add_edge("generate_answer",  END)

    # ── Conditional: after classify_question ──────────────────────────────────
    # sports_news  → retrieve_initial
    # sports_wiki  → wikipedia_search
    # off_topic    → generate_answer (polite decline)
    builder.add_conditional_edges(
        "classify_question",
        manager.route_after_classify,
        {
            "retrieve_initial": "retrieve_initial",
            "wikipedia_search": "wikipedia_search",
            "generate_answer":  "generate_answer",
        },
    )

    # ── Conditional: after check_recency ──────────────────────────────────────
    # RECENT / OLD → generate_answer
    # NOT_FOUND    → wikipedia_search (safe: classify already confirmed sports)
    builder.add_conditional_edges(
        "check_recency",
        manager.route_after_recency_check,
        {
            "generate_answer":  "generate_answer",
            "wikipedia_search": "wikipedia_search",
        },
    )

    return builder.compile()


# Singleton — import once, reuse
_graph = None


def get_graph():
    """Get or create the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph