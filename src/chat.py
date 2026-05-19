import os
from dotenv import load_dotenv, find_dotenv
from langchain_postgres import PGVector
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from util.tools import build_agent_tools
import datetime
import json
from pathlib import Path

COLLECTION_NAME = "news_articles"


def main():
    load_dotenv(find_dotenv(), override=True)
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DB_URL missing in .env")

    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=db_url,
        use_jsonb=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm: ChatNVIDIA = ChatNVIDIA(
        model="nvidia/nemotron-3-super-120b-a12b",
        temperature=0.2,
        max_tokens=16384,
        reasoning_budget=16384,
        chat_template_kwargs={"enable_thinking":True},
    )

    tools = build_agent_tools()

    agent = create_agent(llm, tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Bạn là trợ lý cập nhật tin tức thể thao và bóng đá. "
                "Trả lời ngắn gọn, đúng theo dữ liệu có được, đúng theo sự thật, không được bịa đặt. "
                "\n"
                "=== QUY TRÌNH LẤY THÔNG TIN ===\n"
                "1. Lấy thông tin từ Vector Database RAG (sử dụng retriever trước).\n"
                "2. Nếu thông tin từ RAG đủ và đảm bảo tính mới nhất → trả lời ngay.\n"
                "3. Nếu thông tin từ RAG không đủ hoặc không chắc về tính mới nhất, hãy:\n"
                "   - Gọi tool crawl_news_feed để lấy dữ liệu mới từ RSS.\n"
                "   - Chỉ crawl từ RSS hợp lệ hoặc RSS mặc định: https://vnexpress.net/rss/the-thao.rss\n"
                "   - Sau khi crawl, lấy lại thông tin từ Vector Database (retriever sẽ có dữ liệu mới).\n"
                "   - Trả lời dựa trên dữ liệu mới được insert.\n"
                "\n"
                "=== HẠNG CHẾ ===\n"
                "- Không được tự tạo hoặc đoán RSS feed, URL bài báo, hoặc tên trang web.\n"
                "- Nếu RSS bị chặn, không khả dụng, hãy thông báo người dùng.\n"
                "- Không suy diễn hay bịa đặt nội dung khi không có dữ liệu.",
            ),
            ("human", "Câu hỏi: {question}\n"),
        ]
    )

    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        # docs = retriever.get_relevant_documents(question)
        docs = retriever.invoke(question)

        chain = prompt | agent
        answer = chain.invoke({"question": question})

        # Ensure logs directory exists
        Path("logs").mkdir(parents=True, exist_ok=True)

        # Try to serialize the raw answer; fall back to str()
        try:
            serialized = json.dumps(answer, default=lambda o: getattr(o, "dict", lambda: str(o))(), ensure_ascii=False)
        except Exception:
            try:
                # second attempt: if answer is pydantic-like or has .content
                if hasattr(answer, "content"):
                    serialized = json.dumps({"content": answer.content}, ensure_ascii=False)
                else:
                    serialized = json.dumps(str(answer), ensure_ascii=False)
            except Exception:
                serialized = str(answer)

        try:
            with open("logs/answers.log", "a", encoding="utf-8") as lf:
                lf.write(f"{datetime.datetime.utcnow().isoformat()}\n")
                lf.write(f"QUESTION: {question}\n")
                lf.write(serialized + "\n")
                lf.write("---\n")
        except Exception as e:
            print("Failed to write answer log:", e)
        print("\n--- Answer ---")
        print(answer['messages'][-1].content)

        print("\n--- Sources ---")
        for i, d in enumerate(docs, 1):
            print(f"{i}. {d.metadata}")


if __name__ == "__main__":
    main()
