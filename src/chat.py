"""
chat.py
=======
CLI chatbot dùng LangGraph graph thay cho LangChain AgentExecutor.

Thay đổi so với version cũ:
- Bỏ create_agent / AgentExecutor
- Dùng compiled StateGraph từ graph.py
- State rõ ràng: mỗi turn inject {current_question, messages}
- Log giữ nguyên format cũ
"""

import datetime
import json
import logging
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    load_dotenv(find_dotenv(), override=True)

    from graph import get_graph
    graph = get_graph()
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Giữ message history qua các turns
    message_history: list = []

    print("=== AI Sports News Chatbot ===")
    print("Gõ 'exit' hoặc 'quit' để thoát.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Tạm biệt!")
            break

        # ── Invoke graph ──────────────────────────────────────────────────────
        t_start = time.perf_counter()

        state_input = {
            "current_question": question,
            "messages": message_history + [HumanMessage(content=question)],
        }

        try:
            result = graph.invoke(state_input)
        except Exception as e:
            logger.exception("Graph invoke failed: %s", e)
            print(f"\n[Lỗi] Không thể xử lý câu hỏi: {e}\n")
            continue

        t_elapsed = time.perf_counter() - t_start

        # ── Update history ────────────────────────────────────────────────────
        # Chỉ giữ messages từ kết quả (LangGraph đã merge add_messages)
        message_history = result.get("messages", message_history)

        # ── Output ────────────────────────────────────────────────────────────
        final_answer = result.get("final_answer", "(không có câu trả lời)")

        print(f"\n--- Answer ({t_elapsed:.2f}s) ---")
        print(final_answer)
        print()

        # ── Log ───────────────────────────────────────────────────────────────
        _write_log(
            question=question,
            result=result,
            elapsed=t_elapsed,
        )


def _write_log(question: str, result: dict, elapsed: float) -> None:
    def _serialize(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return str(obj)

    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "question": question,
        "elapsed_seconds": round(elapsed, 3),
        "recency_status": result.get("recency_status"),
        "crawl_count": result.get("crawl_count"),
        "crawl_history": result.get("crawl_history"),
        "docs_found_count": len(result.get("docs_found", [])),
        "final_answer": result.get("final_answer"),
        "messages": [_serialize(m) for m in result.get("messages", [])],
    }

    try:
        with open("logs/answers.log", "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False, indent=2))
            lf.write("\n---\n")
    except Exception as e:
        logger.warning("Failed to write log: %s", e)


if __name__ == "__main__":
    main()