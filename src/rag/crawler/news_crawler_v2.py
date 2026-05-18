from trafilatura import fetch_url, extract
from bs4 import BeautifulSoup
from datetime import datetime
import logging, json, os, re

logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> datetime:
    """
    Parse chuỗi ngày tháng từ VNExpress sang đối tượng datetime.

    Định dạng đầu vào ví dụ: "Thứ Hai, 20/11/2023, 10:00 (GMT+7)"
    Chiến lược: dùng Regex để bỏ qua phần tên thứ bằng tiếng Việt
    và phần timezone, chỉ lấy ngày + giờ để parse.

    Args:
        date_str: Chuỗi ngày tháng thô từ HTML của VNExpress.

    Returns:
        Đối tượng datetime tương ứng, hoặc datetime.min nếu không parse được.
    """
    if not date_str or not date_str.strip():
        return datetime.min

    # Trích xuất phần "dd/mm/yyyy" và "HH:MM" bằng regex,
    # bỏ qua hoàn toàn tên thứ tiếng Việt và timezone (GMT+7).
    match = re.search(r"(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{2}:\d{2})", date_str)

    if not match:
        logger.warning("parse_date: Không nhận diện được định dạng ngày '%s'", date_str)
        return datetime.min

    try:
        date_part = match.group(1)  # "20/11/2023"
        time_part = match.group(2)  # "10:00"
        return datetime.strptime(f"{date_part} {time_part}", "%d/%m/%Y %H:%M")
    except ValueError as e:
        logger.warning("parse_date: Lỗi khi parse ngày '%s': %s", date_str, e)
        return datetime.min


""" Luồng hoạt động: RSSFeedParser -> NewsCrawler -> Storage """


class RSSFeedParser:
    def __init__(self, feed_url: str) -> None:
        self.feed_url = feed_url
        self.document: str = ""

    def get_feed_content(self) -> None:
        logger.info(
            "[RSSFeedParser - get_feed_content()] Fetching RSS Feed: %s", self.feed_url
        )
        self.document: str = fetch_url(self.feed_url)

        if not self.document:
            logger.error(
                "[RSSFeedParser - get_feed_content()] Failed to fetch RSS feed: %s",
                self.feed_url,
            )
            return []

    def parse_urls(self) -> list[str]:
        if self.document is None:
            self.get_feed_content()
        soup: BeautifulSoup = BeautifulSoup(self.document, "xml")
        rss_item: list[str] = soup.find_all("item")
        news_urls: list[str] = []

        for item in rss_item:
            link_tag = item.find("guid") or item.find("link")
            if link_tag and link_tag.text.startswith("http"):
                news_urls.append(link_tag.text)

        logger.info(
            "[RSSFeedParser - parse_urls()] Found %d article URLs", len(news_urls)
        )

        return news_urls


class Storage:
    """
    Kiểm tra bài báo đã tồn tại trong output.json chưa,
    nếu chưa thì thêm vào và lưu lại theo thứ tự mới nhất → cũ nhất.
    """

    def __init__(self, output_url: str, news_data: list[dict]) -> None:
        self.output_url = output_url
        self.news_data = news_data

    def _load_existing_data(self) -> list[dict]:
        """Đọc dữ liệu hiện có từ file JSON. Trả về list rỗng nếu file chưa tồn tại."""
        if not os.path.exists(self.output_url):
            return []
        try:
            with open(self.output_url, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.warning(
                        "[Storage] File %s sai format, tiến hành reset.",
                        self.output_url,
                    )
                    return []
                return data
        except json.JSONDecodeError:
            logger.error(
                "[Storage] File %s bị lỗi JSON, tiến hành reset.", self.output_url
            )
            return []

    def _save_data(self, data: list[dict]) -> None:
        """Ghi danh sách bài báo ra file JSON."""
        with open(self.output_url, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _is_duplicate(self, item: dict, existing_data: list[dict]) -> bool:
        """Kiểm tra xem bài báo đã tồn tại trong danh sách chưa (so sánh theo URL)."""
        item_url = item.get("url")
        return any(existing.get("url") == item_url for existing in existing_data)

    def operate_function(self) -> None:
        data = self._load_existing_data()

        new_count = 0
        for item in self.news_data:
            if not self._is_duplicate(item, data):
                data.append(item)
                new_count += 1

        # Sắp xếp toàn bộ danh sách từ MỚI NHẤT đến CŨ NHẤT
        data.sort(key=lambda x: parse_date(x.get("date", "")), reverse=True)

        self._save_data(data)
        logger.info(
            "[Storage] Hoàn tất: thêm %d bài mới, tổng %d bài. File: %s",
            new_count,
            len(data),
            self.output_url,
        )


class NewsCrawler:
    """Dựa vào các cái link URL xuất hiện trong feed RSS, tiến hành crawl dữ liệu bài báo"""

    def __init__(self, output_url: str, news_urls: list[str]) -> None:
        self.news_urls: list[str] = news_urls
        self.output_url: str = output_url
        self.news_data: list[dict] = []

    def extract_metadata(self, document: str) -> tuple[str, str]:
        soup: BeautifulSoup = BeautifulSoup(document, "html.parser")
        title: str = soup.title.text if soup.title else ""
        date_tag = soup.find("span", class_="date")
        date = date_tag.text if date_tag else ""
        return title, date

    def content_extractor(self, news_url: str) -> None:
        document: str = fetch_url(news_url)

        if document is None:
            print("Nothing inside the document")
            return None

        text: str = extract(document)
        title, date = self.extract_metadata(document)

        self.news_data.append(
            {"title": title, "date": date, "content": text, "url": news_url}
        )

    def news_content_collection(self) -> list[dict]:
        if self.news_urls is None:
            print(f"There are no working url news")
            return []

        for url in self.news_urls:
            self.content_extractor(url)

        return self.news_data


def operation(url: str, output_url: str):
    rss_feed_parser: RSSFeedParser = RSSFeedParser(feed_url=url)
    rss_feed_parser.get_feed_content()
    news_urls: list[str] = rss_feed_parser.parse_urls()

    crawler: NewsCrawler = NewsCrawler(output_url=output_url, news_urls=news_urls)
    news_data: list[dict] = crawler.news_content_collection()

    storage: Storage = Storage(output_url=output_url, news_data=news_data)
    storage.operate_function()
