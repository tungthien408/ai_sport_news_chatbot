"""
graph_nodes.py
==============
OOP-based node implementations for the LangGraph sports news chatbot.

Classes:
    WikipediaSearcher — Encapsulates Wikipedia search with error handling.
    GraphNodeManager  — Contains all graph nodes and routing logic.

Flow:
  process_input → classify_question
      ├─ sports_news  → retrieve_initial → check_recency
      │                                      ├─ RECENT / OLD → generate_answer → END
      │                                      └─ NOT_FOUND    → wikipedia_search → generate_answer → END
      ├─ sports_wiki  → wikipedia_search → generate_answer → END
      └─ off_topic    → generate_answer (polite decline) → END
"""

import logging
import re
from datetime import datetime, timedelta, timezone

import wikipedia
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from graph_state import GraphState
from util.tools import current_time, retrieve_context

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# WikipediaSearcher
# ══════════════════════════════════════════════════════════════════════════════
class WikipediaSearcher:
    """Handles Wikipedia lookups with language detection and error handling.

    Attributes:
        default_sentences: Number of summary sentences to retrieve.
        default_lang: Fallback language code when detection fails.
    """

    _VI_PATTERN = re.compile(
        r"[àáảãạăắằẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
        r"ùúủũụưứừửữựỳýỷỹỵ]",
        re.IGNORECASE,
    )

    def __init__(self, default_sentences: int = 5, default_lang: str = "en") -> None:
        self.default_sentences = default_sentences
        self.default_lang = default_lang

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, sentences: int | None = None) -> str | None:
        """Search Wikipedia for a query and return a summary.

        Tries the detected language first, then falls back to English.
        Handles disambiguation by picking the first suggestion.

        Args:
            query: The search term (e.g., "Cristiano Ronaldo").
            sentences: Number of summary sentences. Defaults to self.default_sentences.

        Returns:
            The Wikipedia summary text, or None if nothing was found.
        """
        sentences = sentences or self.default_sentences
        lang = self._detect_language(query)
        languages_to_try = [lang] if lang == "en" else [lang, "en"]

        for lang_code in languages_to_try:
            result = self._search_in_language(query, lang_code, sentences)
            if result:
                return result

        logger.warning("[WikipediaSearcher] No results for %r in any language", query)
        return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_language(self, text: str) -> str:
        """Detect whether the query is Vietnamese or English."""
        if self._VI_PATTERN.search(text):
            return "vi"
        return self.default_lang

    def _search_in_language(
        self, query: str, lang: str, sentences: int
    ) -> str | None:
        """Attempt a Wikipedia search in a specific language.

        Handles DisambiguationError by retrying with the first option.
        """
        wikipedia.set_lang(lang)

        try:
            summary = wikipedia.summary(query, sentences=sentences)
            page = wikipedia.page(query)
            source_info = f"\n\n📖 Source: {page.url}"
            logger.info(
                "[WikipediaSearcher] Found article in '%s': %s", lang, page.title
            )
            return summary + source_info

        except wikipedia.exceptions.DisambiguationError as e:
            return self._handle_disambiguation(e, lang, sentences)

        except wikipedia.exceptions.PageError:
            logger.info(
                "[WikipediaSearcher] No page found for %r in '%s'", query, lang
            )
            return None

        except Exception as e:
            logger.error(
                "[WikipediaSearcher] Unexpected error for %r in '%s': %s",
                query,
                lang,
                e,
            )
            return None

    def _handle_disambiguation(
        self,
        error: wikipedia.exceptions.DisambiguationError,
        lang: str,
        sentences: int,
    ) -> str | None:
        """Handle disambiguation by trying the first suggested option."""
        options = error.options
        if not options:
            return None

        logger.info(
            "[WikipediaSearcher] Disambiguation for '%s', trying: %s",
            lang,
            options[0],
        )

        try:
            return wikipedia.summary(options[0], sentences=sentences)
        except Exception:
            logger.warning(
                "[WikipediaSearcher] Disambiguation fallback also failed for %r",
                options[0],
            )
            return None


# ══════════════════════════════════════════════════════════════════════════════
# GraphNodeManager
# ══════════════════════════════════════════════════════════════════════════════
class GraphNodeManager:
    """Manages all LangGraph nodes and routing logic for the sports chatbot.

    Each public method is a graph node: it receives GraphState and returns
    a dict of state updates. Routing methods return a node name string.
    """

    # ── System prompts ────────────────────────────────────────────────────────

    GENERATE_SYSTEM = (
        "Bạn là trợ lý tin tức thể thao. Trả lời ngắn gọn, chính xác, "
        "chỉ dựa trên context được cung cấp.\n"
        "Không được bịa đặt hay suy diễn. Luôn kèm citation: tiêu đề bài + URL + ngày.\n"
        "Nếu context không đủ, nói thẳng không có đủ thông tin.\n"
        "Luôn luôn phản hồi với người dùng bằng chính ngôn ngữ mà họ dùng để hỏi bạn.\n"
        "Hãy giữ nguyên tất cả các ngôn từ thể hiện sự không chắc chắn và tính điều kiện "
        "từ nguồn tài liệu chính xác như cách chúng được viết "
        "(ví dụ: 'nếu', 'được báo cáo là', 'được kỳ vọng là', 'có thể'). "
        "Tuyệt đối không bao giờ trình bày các sự thật có điều kiện như thể chúng là "
        "những điều chắc chắn.\n"
        "Hãy tổng hợp thông tin từ TẤT CẢ các tài liệu được cung cấp. "
        "Nếu có nhiều tài liệu cùng giải quyết một câu hỏi, "
        "hãy kết hợp chúng lại thành một câu trả lời toàn diện.\n"
    )

    CLASSIFY_SYSTEM = (
        "You are a sports topic classifier. Given a user question, classify it into "
        "exactly one of three categories:\n\n"
        "1. **sports_news** — The user asks about recent sports events, match results, "
        "transfers, injuries, tournament updates, or any time-sensitive sports information.\n"
        "   Examples: 'Kết quả trận Real Madrid tối qua?', "
        "'Latest Premier League standings', 'Tin chuyển nhượng MU'\n\n"
        "2. **sports_wiki** — The user asks general knowledge about a sports figure, "
        "team, tournament, or concept. Biographical info, career stats, "
        "historical facts, records.\n"
        "   Examples: 'Mourinho là ai?', 'How many goals has Ronaldo scored?', "
        "'Lịch sử World Cup', 'Tell me about Messi'\n\n"
        "3. **off_topic** — The question is NOT related to sports at all. "
        "Food, politics, science, entertainment (non-sport), etc.\n"
        "   Examples: 'Cách nấu phở?', 'Who is the president?', "
        "'Best pizza in town'\n\n"
        "Respond with ONLY the category name: sports_news, sports_wiki, or off_topic.\n"
        "Do not include any other text."
    )

    WIKI_GENERATE_SYSTEM = (
        "Bạn là trợ lý tin tức thể thao. Trả lời dựa trên thông tin Wikipedia "
        "được cung cấp.\n"
        "Tóm tắt thông tin một cách rõ ràng và chính xác. "
        "Ghi rõ nguồn từ Wikipedia.\n"
        "Luôn luôn phản hồi với người dùng bằng chính ngôn ngữ mà họ dùng để hỏi bạn.\n"
    )

    OFF_TOPIC_RESPONSE_VI = (
        "Xin lỗi, tôi là trợ lý tin tức thể thao và chỉ có thể hỗ trợ các câu hỏi "
        "liên quan đến thể thao. Vui lòng hỏi về tin tức thể thao, cầu thủ, đội bóng, "
        "hoặc các giải đấu!"
    )

    OFF_TOPIC_RESPONSE_EN = (
        "Sorry, I'm a sports news assistant and can only help with sports-related "
        "questions. Please ask about sports news, players, teams, or tournaments!"
    )

    DEFAULT_CRAWL_URLS = [
        "https://vnexpress.net/rss/the-thao.rss",
        "https://feeds.bbci.co.uk/sport/rss.xml",
    ]

    def __init__(self) -> None:
        self._llm = ChatNVIDIA(
            model="nvidia/nemotron-3-super-120b-a12b",
            temperature=0.2,
        )
        self._wiki_searcher = WikipediaSearcher(default_sentences=5)

    # ══════════════════════════════════════════════════════════════════════════
    # Node 1: process_input
    # ══════════════════════════════════════════════════════════════════════════

    def process_input(self, state: GraphState) -> dict:
        """Parse user input, extract time preference, init tracking fields."""
        question = state["current_question"]
        time_hours = self._extract_time_preference(question)

        logger.info("[process_input] question=%r time_pref=%dh", question, time_hours)

        return {
            "time_preference_hours": time_hours,
            "question_type": "sports_news",  # default, overridden by classify
            "crawl_count": 0,
            "crawl_history": [],
            "user_feedback": None,
            "recency_status": "NOT_FOUND",
            "docs_found": [],
            "wiki_content": None,
            "final_answer": "",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Node 2: classify_question (Sports Gatekeeper + Router)
    # ══════════════════════════════════════════════════════════════════════════

    def classify_question(self, state: GraphState) -> dict:
        """Classify the question as sports_news, sports_wiki, or off_topic.

        This node acts as a gatekeeper: off-topic questions are rejected
        here and never reach the vector DB or Wikipedia.
        """
        question = state["current_question"]

        messages = [
            SystemMessage(content=self.CLASSIFY_SYSTEM),
            HumanMessage(content=question),
        ]

        response = self._llm.invoke(messages)
        raw_type = response.content.strip().lower()

        # Validate the classification — default to sports_news if unexpected
        valid_types = {"sports_news", "sports_wiki", "off_topic"}
        question_type = raw_type if raw_type in valid_types else "sports_news"

        logger.info(
            "[classify_question] question=%r → %s (raw=%r)",
            question,
            question_type,
            raw_type,
        )

        return {"question_type": question_type}

    # ══════════════════════════════════════════════════════════════════════════
    # Node 3: retrieve_initial
    # ══════════════════════════════════════════════════════════════════════════

    def retrieve_initial(self, state: GraphState) -> dict:
        """Query vector DB and populate docs_found."""
        question = state["current_question"]
        hours = state["time_preference_hours"]
        result_str = retrieve_context(question, hours_ago=hours)
        docs = self._parse_retrieve_result(result_str)

        logger.info("[retrieve_initial] found %d docs", len(docs))
        return {"docs_found": docs}

    # ══════════════════════════════════════════════════════════════════════════
    # Node 4: check_recency
    # ══════════════════════════════════════════════════════════════════════════

    def check_recency(self, state: GraphState) -> dict:
        """Check whether retrieved docs fall within the requested time window."""
        docs: list[Document] = state["docs_found"]
        hours: int = state["time_preference_hours"]

        if not docs:
            logger.info("[check_recency] no docs → NOT_FOUND")
            return {"recency_status": "NOT_FOUND"}

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)

        for doc in docs:
            pub_str = doc.metadata.get("published_at") or doc.metadata.get("date", "")
            if not pub_str:
                continue
            try:
                pub_dt = datetime.fromisoformat(pub_str)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                if pub_dt >= cutoff:
                    logger.info(
                        "[check_recency] RECENT doc found: %s",
                        doc.metadata.get("title"),
                    )
                    return {"recency_status": "RECENT"}
            except (ValueError, TypeError):
                continue

        logger.info(
            "[check_recency] all docs are OLD (cutoff=%s)", cutoff.isoformat()
        )
        return {"recency_status": "OLD"}

    # ══════════════════════════════════════════════════════════════════════════
    # Node 5: wikipedia_search
    # ══════════════════════════════════════════════════════════════════════════

    def wikipedia_search(self, state: GraphState) -> dict:
        """Search Wikipedia for the user's question and store results.

        This node is only reachable for sports-classified questions
        (either directly via sports_wiki, or as a NOT_FOUND fallback
        from the vector DB path).
        """
        question = state["current_question"]
        logger.info("[wikipedia_search] searching for: %r", question)

        wiki_content = self._wiki_searcher.search(question)

        if not wiki_content:
            logger.info("[wikipedia_search] no Wikipedia results found")
            return {"wiki_content": None}

        # Create a Document so generate_answer can handle it uniformly
        wiki_doc = Document(
            page_content=wiki_content,
            metadata={
                "title": f"Wikipedia: {question}",
                "url": self._extract_wiki_url(wiki_content),
                "date": datetime.now(timezone.utc).isoformat(),
                "source": "wikipedia",
            },
        )

        logger.info("[wikipedia_search] found Wikipedia content (%d chars)", len(wiki_content))
        return {
            "wiki_content": wiki_content,
            "docs_found": [wiki_doc],
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Node 6: generate_answer
    # ══════════════════════════════════════════════════════════════════════════

    def generate_answer(self, state: GraphState) -> dict:
        """Generate the final answer based on question type and available data.

        Handles three scenarios:
        - off_topic: returns a polite decline message.
        - sports_wiki (with wiki_content): synthesizes from Wikipedia.
        - sports_news (with docs_found): synthesizes from vector DB docs.
        - No data available: returns a "no info found" message.
        """
        question = state["current_question"]
        question_type = state.get("question_type", "sports_news")
        docs: list[Document] = state.get("docs_found", [])
        wiki_content: str | None = state.get("wiki_content")

        # ── Off-topic: polite decline ─────────────────────────────────────────
        if question_type == "off_topic":
            answer = self._get_off_topic_response(question)
            return {
                "final_answer": answer,
                "messages": [AIMessage(content=answer)],
            }

        # ── Wikipedia content available ───────────────────────────────────────
        if wiki_content:
            return self._generate_wiki_answer(question, wiki_content)

        # ── Vector DB docs available ──────────────────────────────────────────
        if docs:
            return self._generate_news_answer(question, state)

        # ── Nothing found at all ──────────────────────────────────────────────
        answer = "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu hoặc Wikipedia."
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Routing functions
    # ══════════════════════════════════════════════════════════════════════════

    def route_after_classify(self, state: GraphState) -> str:
        """Route after classify_question node.

        - sports_news  → retrieve_initial (vector DB path)
        - sports_wiki  → wikipedia_search
        - off_topic    → generate_answer (polite decline)
        """
        question_type = state.get("question_type", "sports_news")

        if question_type == "sports_wiki":
            return "wikipedia_search"
        if question_type == "off_topic":
            return "generate_answer"
        return "retrieve_initial"

    def route_after_recency_check(self, state: GraphState) -> str:
        """Route after check_recency node.

        - RECENT / OLD → generate_answer (use what we found)
        - NOT_FOUND    → wikipedia_search (safe: classify already confirmed sports)
        """
        status = state.get("recency_status", "NOT_FOUND")

        if status in ("RECENT", "OLD"):
            return "generate_answer"

        # NOT_FOUND → fallback to Wikipedia (safe because classify
        # already confirmed this is a sports-related question)
        logger.info(
            "[route_after_recency_check] NOT_FOUND → falling back to Wikipedia"
        )
        return "wikipedia_search"

    # ══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _generate_wiki_answer(self, question: str, wiki_content: str) -> dict:
        """Generate an answer using Wikipedia content."""
        user_prompt = (
            f"Câu hỏi: {question}\n\n"
            f"Thông tin từ Wikipedia:\n{wiki_content}\n\n"
            "Hãy trả lời câu hỏi dựa trên thông tin Wikipedia trên. "
            "Tóm tắt ngắn gọn và ghi rõ nguồn từ Wikipedia."
        )

        messages = [
            SystemMessage(content=self.WIKI_GENERATE_SYSTEM),
            HumanMessage(content=user_prompt),
        ]

        logger.info("[generate_answer] calling LLM with Wikipedia context")
        response = self._llm.invoke(messages)
        answer_text = response.content

        return {
            "final_answer": answer_text,
            "messages": [AIMessage(content=answer_text)],
        }

    def _generate_news_answer(self, question: str, state: GraphState) -> dict:
        """Generate an answer using vector DB documents."""
        docs: list[Document] = state.get("docs_found", [])
        recency_status = state.get("recency_status", "NOT_FOUND")

        context_parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            recency_warn = meta.get("recency_warning", "")
            warn_text = f"\n    ⚠️  {recency_warn}" if recency_warn else ""
            context_parts.append(
                f"[{i}] Tiêu đề: {meta.get('title', 'N/A')}\n"
                f"    URL: {meta.get('url', 'N/A')}\n"
                f"    Ngày: {meta.get('date', 'N/A')}{warn_text}\n"
                f"    Nội dung: {doc.page_content[:600]}"
            )
        context_str = "\n\n".join(context_parts)

        recency_note = ""
        if recency_status == "OLD":
            recency_note = (
                "\n⚠️  LƯU Ý: Các tài liệu tìm được đều cũ hơn khung thời gian "
                "yêu cầu. Hãy nêu rõ đây là thông tin lịch sử và ghi ngày của bài."
            )

        user_prompt = (
            f"Câu hỏi: {question}\n\n"
            f"Context từ database:{recency_note}\n\n"
            f"{context_str}\n\n"
            "Hãy trả lời câu hỏi dựa trên context trên. "
            "Tóm tắt 1-2 câu, sau đó list nguồn (tiêu đề + URL + ngày)."
        )

        messages = [
            SystemMessage(content=self.GENERATE_SYSTEM),
            HumanMessage(content=user_prompt),
        ]

        logger.info(
            "[generate_answer] calling LLM, recency_status=%s", recency_status
        )
        response = self._llm.invoke(messages)
        answer_text = response.content

        return {
            "final_answer": answer_text,
            "messages": [AIMessage(content=answer_text)],
        }

    def _get_off_topic_response(self, question: str) -> str:
        """Return the appropriate off-topic decline message based on language."""
        if WikipediaSearcher._VI_PATTERN.search(question):
            return self.OFF_TOPIC_RESPONSE_VI
        return self.OFF_TOPIC_RESPONSE_EN

    @staticmethod
    def _extract_time_preference(question: str) -> int:
        """Extract time preference from the question. Default = 12h."""
        q = question.lower()

        if re.search(r"tháng này|trong tháng", q):
            return 720  # ~30 days

        if re.search(r"tuần này|trong tuần|7 ngày", q):
            return 168

        if re.search(r"hôm nay|trong ngày", q):
            return 24

        # "trong X giờ" or "Xh"
        match = re.search(r"trong\s*(\d+)\s*giờ", q) or re.search(
            r"(\d+)\s*h\b", q
        )
        if match:
            return int(match.group(1))

        # "mới nhất", "gần nhất" → default 12h
        return 12

    @staticmethod
    def _extract_rss_url_from_question(question: str) -> str | None:
        """Extract an RSS URL provided by the user in their question."""
        match = re.search(r"https?://\S+\.rss\b", question, re.IGNORECASE)
        if match:
            return match.group(0)
        return None

    @staticmethod
    def _parse_retrieve_result(result: str) -> list[Document]:
        """Convert the string returned by retrieve_context() into Documents."""
        if not result or "Không tìm thấy" in result or "Lỗi khi query" in result:
            return []

        blocks = [b.strip() for b in result.split("\n\n") if b.strip()]
        docs: list[Document] = []

        for blk in blocks:
            title_m = re.search(r"Nguồn:\s*(.*)", blk)
            url_m = re.search(r"URL:\s*(.*)", blk)
            date_m = re.search(r"Ngày:\s*(.*)", blk)
            content_m = re.search(r"Nội dung:\s*(.*)", blk, re.DOTALL)

            docs.append(
                Document(
                    page_content=content_m.group(1).strip() if content_m else "",
                    metadata={
                        "title": title_m.group(1).strip() if title_m else "",
                        "url": url_m.group(1).strip() if url_m else "",
                        "date": date_m.group(1).strip() if date_m else "",
                    },
                )
            )

        return docs

    @staticmethod
    def _extract_wiki_url(wiki_content: str) -> str:
        """Extract the Wikipedia URL from the content footer."""
        match = re.search(r"(https?://\S+wikipedia\S+)", wiki_content)
        return match.group(1) if match else "https://wikipedia.org"
