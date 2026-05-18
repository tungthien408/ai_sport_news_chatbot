"""
Vietstock Industry News Scraper
================================
Lấy tin tức theo ngành từ finance.vietstock.vn/Data/GetIndustryNews

Fix so với code cũ:
- Lấy __RequestVerificationToken từ HTML body (không phải Set-Cookie header)
- Thêm đầy đủ headers để server nhận ra XHR request
- Cache session để gọi nhiều ngành mà không phải handshake lại
"""

import requests
import json
import time
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────
# Danh sách ngành Vietstock (industry_id: tên)
# ──────────────────────────────────────────────
INDUSTRY_MAP = {
    10: "nang-luong",       # Năng lượng
    2:  "ngan-hang",        # Ngân hàng
    3:  "bat-dong-san",     # Bất động sản
    4:  "cong-nghe",        # Công nghệ
    5:  "thuc-pham",        # Thực phẩm & Đồ uống
    # Thêm các ngành khác tại đây
}


class VietstockNewsScraper:
    BASE_URL = "https://finance.vietstock.vn"
    NEWS_API  = f"{BASE_URL}/Data/GetIndustryNews"

    # Headers giả lập browser — quan trọng để server trả JSON thay vì HTML
    HEADERS = {
        "Accept":            "application/json, text/javascript, */*; q=0.01",   # ← báo server muốn JSON
        "Accept-Encoding":   "gzip, deflate, br",
        "Accept-Language":   "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection":        "keep-alive",
        "Content-Type":      "application/x-www-form-urlencoded; charset=UTF-8",
        "User-Agent":        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "X-Requested-With":  "XMLHttpRequest",                                    # ← đánh dấu đây là AJAX
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._csrf_token: str | None = None        # cache token — dùng lại được trong cùng 1 session

    # ──────────────────────────────────────────
    # Bước 1: Lấy session cookies + CSRF token
    # ──────────────────────────────────────────
    def _init_session(self, industry_id: int) -> str:
        """
        GET trang ngành để:
        - Tạo ASP.NET_SessionId (cookie)
        - Lấy __RequestVerificationToken từ HTML body (hidden input)
        
        ⚠️  Token trong cookie ≠ token trong form body.
            Server ASP.NET yêu cầu cả hai phải khớp.
            Code cũ lấy nhầm từ Set-Cookie → server trả HTML thay vì JSON.
        """
        slug = INDUSTRY_MAP.get(industry_id, "nang-luong")
        page_url = f"{self.BASE_URL}/nganh/{industry_id}-{slug}.htm"
        print("\n\n ====== \n\n")
        print(page_url)
        print("\n\n ====== \n\n")

        self.session.headers.update({"Referer": page_url})

        resp = self.session.get(page_url, timeout=15)
        print(self.session.cookies.get_dict())
        resp.raise_for_status()

        # Parse token từ HTML — đây là FIX chính
        soup = BeautifulSoup(resp.text, "html.parser")
        token_input = soup.find("input", {"name": "__RequestVerificationToken"})

        if not token_input:
            # Fallback: thử tìm trong meta tag (một số trang ASP.NET dùng cách này)
            meta = soup.find("meta", {"name": "csrf-token"})
            if meta:
                return meta["content"]
            raise RuntimeError(
                "❌ Không tìm thấy __RequestVerificationToken trong HTML.\n"
                "   Có thể trang đã thay đổi cấu trúc hoặc bị chặn."
            )

        token = token_input["value"]
        print(f"✅ Token lấy được: {token[:20]}...")
        print("COOKIE:", self.session.cookies.get("__RequestVerificationToken"))
        print("FORM:", token)
        return token

    # ──────────────────────────────────────────
    # Bước 2: Gọi API lấy tin tức
    # ──────────────────────────────────────────
    def get_industry_news(
        self,
        industry_id: int = 10,
        page: int = 1,
        page_size: int = 20,
    ) -> list[dict]:
        """
        Lấy tin tức theo ngành.

        Args:
            industry_id: ID ngành (xem INDUSTRY_MAP)
            page:        Trang (bắt đầu từ 1)
            page_size:   Số bài mỗi trang (mặc định 20)

        Returns:
            Danh sách bài báo (list of dict)
        """
        # Lấy token lần đầu, cache lại cho các lần gọi sau
        if not self._csrf_token:
            self._csrf_token = self._init_session(industry_id)

        payload = {
            "industryId":                 str(industry_id),
            "page":                       str(page),
            "pageSize":                   str(page_size),
            "__RequestVerificationToken": self._csrf_token,
        }

        resp = self.session.post(self.NEWS_API, data=payload, timeout=15)
        resp.raise_for_status()

        # Kiểm tra server có trả JSON không
        content_type = resp.headers.get("Content-Type", "")
        if "json" not in content_type:
            raise RuntimeError(
                f"❌ Server trả {content_type} thay vì JSON.\n"
                f"   Nội dung đầu: {resp.text[:200]}"
            )

        raw = resp.content.decode("utf-8-sig")   # xử lý BOM nếu có
        return json.loads(raw)

    # ──────────────────────────────────────────
    # Tối ưu: lấy nhiều ngành song song
    # ──────────────────────────────────────────
    def get_multiple_industries(
        self,
        industry_ids: list[int],
        page: int = 1,
        delay: float = 1,           # delay giữa các request (giây) — tránh bị rate-limit
    ) -> dict[int, list[dict]]:
        """
        Lấy tin tức nhiều ngành tuần tự, dùng lại 1 session duy nhất.
        Nhanh hơn nhiều so với tạo session mới cho mỗi ngành.

        delay=0.5 là giá trị an toàn — giảm xuống 0.2 nếu muốn nhanh hơn, tăng lên 1-2 nếu không muốn bị đi tù.
        """
        results: dict[int, list[dict]] = {}

        for i, ind_id in enumerate(industry_ids):
            print(f"📥 Đang lấy ngành {ind_id} ({i+1}/{len(industry_ids)})...")
            try:
                # Reset token khi chuyển ngành (Referer thay đổi)
                self._csrf_token = None
                results[ind_id] = self.get_industry_news(ind_id, page)
                print(f"   ✅ {len(results[ind_id])} bài")
            except Exception as e:
                print(f"   ⚠️  Lỗi ngành {ind_id}: {e}")
                results[ind_id] = []

            if i < len(industry_ids) - 1:
                time.sleep(delay)

        return results


# ──────────────────────────────────────────────
# Chạy thử
# ──────────────────────────────────────────────
if __name__ == "__main__":
    scraper = VietstockNewsScraper()

    # --- Test 1 ngành ---
    print("=== Test 1 ngành (Năng lượng - ID 10) ===")
    try:
        news = scraper.get_industry_news(industry_id=10, page=1)
        print(f"📦 Nhận được {len(news)} bài")
        # Ghi kết quả vào file output.txt
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(f"📦 Nhận được {len(news)} bài\n")
            for i in range(len(news)):
                if news:
                    f.write(json.dumps(news[i], ensure_ascii=False, indent=2) + "\n")
                else:
                    f.write("Không có bài nào\n")
        print("✅ Kết quả đã được ghi vào file output.txt")
    except Exception as e:
        print(f"⚠️  Lỗi: {e}")

    # --- Test nhiều ngành ---
    # print("\n=== Test nhiều ngành ===")
    # all_news = scraper.get_multiple_industries([10, 2, 3])
    # for ind_id, articles in all_news.items():
    #     print(f"Ngành {ind_id}: {len(articles)} bài")

