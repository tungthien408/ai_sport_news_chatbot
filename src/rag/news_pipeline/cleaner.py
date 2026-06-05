from unicodedata import normalize
import logging, re, json, os

logger = logging.getLogger(__name__)


class TextCleaner:
    TABLE_ROW = re.compile(r"^\|.+\|$", re.MULTILINE)
    BYLINE_PATTERN = re.compile(
        r"\n[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][^\n]{0,80}"
        r"(?:\s*\((?:theo|tổng hợp|ghi|báo)[^\)]{0,60}\))?\.?\s*$",
        re.IGNORECASE,
    )

    def __init__(self, output_url: str) -> None:
        self.text: str = ""
        self.output_url = output_url

    def load_existing_data(self) -> list[dict]:
        """Đọc dữ liệu hiện có từ file JSON. Trả về list rỗng nếu file chưa tồn tại."""
        if not os.path.exists(self.output_url):
            return []

        try:
            with open(self.output_url, "r", encoding="utf-8") as f:
                data = json.load(f)

                if not isinstance(data, list):
                    logger.warning("[Storage] File %s sai format.", self.output_url)
                    return []

                return data
        except json.JSONDecodeError:
            logger.error("[Storage] File %s bị lỗi JSON.", self.output_url)
            return []

    def unicode_normalization(self) -> None:
        self.text: str = normalize("NFC", self.text)

    def zero_width_removal(self) -> None:
        self.text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", self.text)

    def clean_tables(self):
        """Chuyển dòng table markdown thành văn bản phẳng dễ embed hơn."""

        # Giữ lại nội dung cell nhưng bỏ pipes
        def _row_to_text(m: re.Match) -> None:
            cells = [c.strip() for c in m.group(0).split("|") if c.strip()]
            return " | ".join(cells)

        self.text = self.TABLE_ROW.sub(_row_to_text, self.text)

    def space_normalization(self) -> None:
        self.text = re.sub(
            r"\n{3,}", "\n\n", self.text
        )  # Tối đa 1 dòng trắng giữa các đoạn
        self.text = re.sub(r"[\t\r]", " ", self.text)  # Tab và \r về space
        self.text = self.text.strip()

    def remove_byline(self) -> None:
        """Loại bỏ tên phóng viên ở cuối bài."""
        self.text = self.BYLINE_PATTERN.sub("", self.text)

    def operation(self) -> list[dict]:
        news_data: list[dict] = self.load_existing_data()

        if not news_data:
            logger.error("[TextCleaner - operation()] Không có bài báo")
            return []

        cleaned: list[dict] = []
        for item in news_data:
            raw_content = item.get("content", "")
            if not raw_content or len(raw_content.strip()) == 0:
                logger.warning(
                    "[TextCleaner] Bỏ qua bài không có nội dung: %s",
                    item.get("title", "N/A"),
                )
                continue  # ← Skip bài lỗi, giữ lại bài khác

            self.text = raw_content
            self.unicode_normalization()
            self.zero_width_removal()
            self.clean_tables()
            self.space_normalization()
            self.remove_byline()

            item["content"] = self.text
            cleaned.append(item)

        return cleaned
