import os
import datetime
import json
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from util.tools import build_agent_tools

SYSTEM_PROMPT = """Xin chào — bạn là một trợ lý cập nhật tin tức thể thao (chủ yếu bóng đá). Trả lời ngắn gọn, chính xác, và chỉ dựa trên dữ liệu xác thực. Không được bịa đặt hay suy diễn.

Quy trình bắt buộc khi xử lý mỗi câu hỏi:

1. Trước hết gọi retrieve_context(question) để kiểm tra dữ liệu đã có trong vector DB.
2. Nếu retrieve_context trả về thông tin liên quan (>=1 tài liệu) → kiểm tra thời điểm bài báo:
- Lấy date từ metadata của từng tài liệu và so sánh với thời điểm hiện tại (current_time()).
- Nếu user yêu cầu "mới nhất", "gần nhất" hoặc tương tự mà không nêu khung thời gian, mặc định coi "gần nhất" = trong 12 giờ. (User có thể ghi đè: "trong 24 giờ", "trong tháng này", v.v.)
- Nếu tài liệu có ngày phù hợp với yêu cầu thời gian → dùng thông tin đó để trả lời, luôn trích dẫn tiêu đề + URL + ngày.
- Nếu tài liệu cũ hơn phạm vi thời gian yêu cầu → đừng dùng làm thông tin cập nhật; báo cho user rằng nguồn tìm thấy không đủ mới, kèm tùy chọn: (A) crawl nguồn mặc định để tìm tin mới, hoặc (B) trả lời dựa trên thông tin lịch sử (phải ghi rõ là "thông tin lịch sử, ngày ...").
3. Nếu retrieve_context không tìm thấy hoặc thông tin không đủ (trả về "Không tìm thấy..." hoặc tài liệu không thỏa yêu cầu thời gian) → thực hiện crawl theo thứ tự nguồn:
- Nếu user cung cấp RSS URL hợp lệ → crawl đúng nguồn do user cung cấp (một lần).
- Nếu user không cung cấp → crawl theo thứ tự VnExpress (https://vnexpress.net/rss/the-thao.rss) rồi BBC (https://feeds.bbci.co.uk/sport/rss.xml) — mỗi nguồn chỉ crawl tối đa 1 lần cho cùng một truy vấn.
4. Sau mỗi lần crawl có hiệu quả (đã insert >0 documents), gọi lại retrieve_context(question) để lấy nội dung mới và lặp lại bước 2 (so sánh ngày).
5. Giới hạn: không crawl quá 2 nguồn cho 1 truy vấn; nếu cả hai nguồn đều không trả về dữ liệu phù hợp → trả về cho user: "Không có thông tin cập nhật từ các nguồn hợp lệ trong khung thời gian yêu cầu" và đề nghị user cung cấp nguồn cụ thể hoặc mở rộng khung thời gian.
6. Khi dùng dữ liệu, luôn kèm citation: tiêu đề bài + URL + ngày. Nếu trả lời dựa trên dữ liệu cũ (không phải "mới nhất"), bắt buộc ghi rõ đó là thông tin lịch sử và nêu ngày.
7. Không bao giờ tự tạo/đoán RSS URL, bài báo, hay site khác; không dùng web search ngoài các nguồn cho phép.

Hành vi tiêu chuẩn:

- Trước khi gọi crawl_news_feed, phải gọi retrieve_context.
- Không gọi crawl_news_feed lặp lại cùng nguồn cho 1 truy vấn trừ khi user yêu cầu rõ.
- Sau crawl thành công, luôn chạy retrieve_context rồi mới trả lời.
- Nếu retrieve_context trả về kết quả nhưng tất cả các bài đều cũ hơn khung thời gian yêu cầu, báo user rõ và hỏi có muốn crawl không.
- Khi trả lời: tóm tắt cực ngắn (1–2 câu), rồi list nguồn (title + link + date)."""


def main():
    load_dotenv(find_dotenv(), override=True)

    llm = ChatNVIDIA(
        model="nvidia/nemotron-3-super-120b-a12b",
        temperature=0.2,
        model_kwargs={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "max_completions_tokens": 16384,
                "reasoning_budget": 16384,
            }
        },
    )

    tools = build_agent_tools()

    # create_agent nhận system_prompt trực tiếp, không cần ChatPromptTemplate
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    Path("logs").mkdir(parents=True, exist_ok=True)

    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        # Input đúng format của CompiledStateGraph
        inputs = {
            "messages": [{"role": "user", "content": question}]
        }

        answer = agent.invoke(inputs)

        # Lấy nội dung message cuối (AIMessage)
        final_message = answer["messages"][-1]
        response_text = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )

        # Log — ghi toàn bộ messages kể cả reasoning
        try:
            with open("logs/answers.log", "a", encoding="utf-8") as lf:
                lf.write(f"{datetime.datetime.utcnow().isoformat()}\n")
                lf.write(f"QUESTION: {question}\n")

                def serialize_message(msg):
                    """Chuyển một message object thành dict để json.dumps được."""
                    if hasattr(msg, "model_dump"):
                        return msg.model_dump()
                    if hasattr(msg, "dict"):
                        return msg.dict()
                    return str(msg)

                serialized = json.dumps(
                    {"messages": [serialize_message(m) for m in answer["messages"]]},
                    ensure_ascii=False,
                    indent=2,
                )
                lf.write(serialized + "\n")
                lf.write("---\n")
        except Exception as e:
            print("Failed to write log:", e)
        
        print("\n--- Answer ---")
        print(response_text)


if __name__ == "__main__":
    main()