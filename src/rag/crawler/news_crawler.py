import feedparser
import json, asyncio 
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from langchain_nvidia_ai_endpoints import ChatNVIDIA


class AnalysisOutput(BaseModel):
    summary: str = Field(description="Tóm tắt ngắn gọn dưới 150-250 từ bằng tiếng Việt")
    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(description="Giá trị từ 0.0 đến 1.0", ge=0.0, le=1.0)
    key_events: List[str]
    sources: List[str]

class CrawlerError(Exception):
    pass

load_dotenv()

RSS: dict = {
    "tai-chinh": "https://vneconomy.vn/tai-chinh.rss",
    "cong-nghiep":  "https://vneconomy.vn/thi-truong-cong-nghiep.rss",
    "kinh-te-xanh": "https://vneconomy.vn/kinh-te-xanh.rss",
}

HEADERS: dict = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

VALID_INDUSTRIES = set(RSS.keys())
llm: ChatNVIDIA = ChatNVIDIA(model="qwen/qwen3.5-122b-a10b")
llm_with_schema = llm.with_structured_output(AnalysisOutput)

def fetch_rss(industry: str, limit: int = 5) -> list[dict]:
    """
    Docstring for fetch_rss
    
    :param industry: Description
    :type industry: str
    :param limit: Description
    :type limit: int
    :return: Description
    :rtype: list[dict]
    """
    if industry not in VALID_INDUSTRIES:
        raise ValueError(f"Lĩnh vực '{industry}' không hợp lệ. Chọn: {VALID_INDUSTRIES}")
    url = RSS.get(industry)
    feed = feedparser.parse(url)

    return [
        {
            "title": e.get("title", ""),
            "url": e.get("link", ""),
            "published": e.get("published", ""),
            "description": e.get("description", "")
        }
        for e in feed.entries[:limit]
    ]

async def extract_fulltext_async(client: httpx.AsyncClient, url: str) -> Optional[dict]:
    """
    Docstring for extract_fulltext_async
    
    :param client: Description
    :type client: httpx.AsyncClient
    :param url: Description
    :type url: str
    :return: Description
    :rtype: dict | None
    """
    try:
        r = await client.get(url, headers=HEADERS, timeout=10, follow_redirects=True)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        # 1. Title
        title_tag = soup.find("h1", class_="name-detail")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # 2. Published time
        published_tag = soup.find("p", class_="date", attrs={"data-field": "distributionDate"})
        published = published_tag.get_text(strip=True) if published_tag else ""

        # 3. Author
        author_tag = soup.find("p", class_="name", attrs={"data-field": "author"})
        author = author_tag.get_text(strip=True) if author_tag else ""

        # 4. Main content
        content_div = soup.find("div", attrs={"data-field": "body"})
        content = ""

        if content_div:
            paragraphs = content_div.find_all("p", class_="text-justify")
            if paragraphs:
                content = "\n\n".join([p.get_text(strip=True) for p in paragraphs])
            else:
                all_p = content_div.find_all("p")
                content = "\n\n".join([p.get_text(strip=True) for p in all_p])

        if not content:
            print("[extract_fulltext_async] Lỗi không có content gì đó nè")
            return None

        return {
            "title": title,
            "published": published,
            "author": author,
            "body": content,
            "url": str(r.url)
        }
    except Exception as e:
        print(f"[extract_fulltext] Error {url}: {e}")
        return None

def build_prompt(articles: list[dict], ticker: str = "") -> str:
    n = len(articles)

    if n <= 3:
        summary_length = "80-100 từ"
    elif n <= 6:
        summary_length = "150-200 từ"
    else:
        summary_length = "250-300 từ"

    combined = "\n\n---\n\n".join(
        f"[Bài {i+1}: {a['title']}]\nNguồn: {a.get('url','')}\nNgày: {a.get('published','')}\n{a.get('body', '')[:3000]}"
        for i, a in enumerate(articles)
    )

    return f"""
        Bạn là chuyên gia phân tích tài chính. Dưới đây là {n} bài báo tài chính liên quan đến {ticker or 'thị trường'}.
        {combined}

        Yêu cầu:
            - summary: tóm tắt {summary_length}, nêu rõ các điểm chính và xu hướng thị trường
            - sentiment: đánh giá tổng thể (bullish/bearish/neutral). Nếu các bài mâu thuẫn nhau, chọn neutral và giải thích trong summary
            - confidence: mức độ chắc chắn của đánh giá (0.0-1.0)
            - key_events: liệt kê 3-5 sự kiện quan trọng nhất (mỗi sự kiện 1 câu ngắn)
            - sources: tiêu đề các bài báo đã dùng

            Chỉ trả về JSON thuần, KHÔNG dùng markdown hay backtick.
    """

def summarize_with_llm(articles: list[dict], ticker: str = "") -> Optional[dict]:
    """
    Docstring for summarize_with_llm
    
    :param articles: Description
    :type articles: list[dict]
    :param ticker: Description
    :type ticker: str
    :return: Description
    :rtype: dict | None
    """
    if not articles:
        print("[summarize_with_llm] Lỗi không có articles nào đó nè")
        return None

    prompt = build_prompt(articles, ticker)

    try:
        result = llm_with_schema.invoke(prompt)
        return result.model_dump()
    except Exception as e:
        print(f"[summarize_with_llm] Fail: {e}")
        return None

async def run_pipeline_async(industry: str, ticker: str = "", limit: int = 5) -> Optional[dict]:
    """Hàm chính để gọi từ API endpoint sau này.
    
    """
    articles_raw = fetch_rss(industry, limit=limit)
    
    async with httpx.AsyncClient(headers=HEADERS) as client:
        tasks = [extract_fulltext_async(client, item["url"]) for item in articles_raw]
        results = await asyncio.gather(*tasks)

    articles = [
        {**raw, **full}
        for raw, full in zip(articles_raw, results)
        if full is not None
    ]

    if not articles:
        print("[run_pipeline_async] Không có articles gì đó nè")
        return None

    return summarize_with_llm(articles, ticker)

# Dễ dàng wrap thành FastAPI sau này:
# @app.get("/news-summary/{industry}")
# async def news_summary(industry: str, ticker: str = ""):
#     result = await asyncio.to_thread(run_pipeline, industry, ticker)
#     if result is None:
#         raise HTTPException(status_code=500, detail="Pipeline thất bại")
#     return result

if __name__ == "__main__":
    result = asyncio.run(run_pipeline_async("cong-nghiep", ticker="NVN", limit=5))
    if result:
        # print("\n=== KẾT QUẢ PHÂN TÍCH ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))
