import logging
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import find_dotenv, load_dotenv

from rag.rag_update_news import rag_update_news

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

FEED_URLS = [
    "https://vnexpress.net/rss/the-thao.rss",
    "https://feeds.bbci.co.uk/sport/rss.xml",
    "https://api.foxsports.com/v2/content/optimized-rss?partnerKey=MB0Wehpmuj2lUhuRhQaafhBjAJqaPU244mlTDK1i",
]


def crawl_feed(feed_url: str) -> int:
    """Crawl one RSS feed and insert results into the vector DB."""
    try:
        logger.info("Starting crawl for %s", feed_url)
        num_inserted = rag_update_news(feed_url)
        logger.info("Finished crawl for %s: inserted %d documents", feed_url, num_inserted)
        return num_inserted
    except Exception as exc:
        logger.exception("Crawl failed for %s: %s", feed_url, exc)
        return 0


def crawl_all_feeds() -> None:
    """Run crawl for every configured feed."""
    Path("logs").mkdir(parents=True, exist_ok=True)

    total = 0
    for feed_url in FEED_URLS:
        if not feed_url:
            continue
        total += crawl_feed(feed_url)

    logger.info("Crawl job complete. Total inserted: %d", total)


def main() -> None:
    load_dotenv(find_dotenv(), override=True)

    scheduler = BlockingScheduler()
    scheduler.add_job(crawl_all_feeds, "interval", hours=1, next_run_time=None)

    logger.info("Starting crawl scheduler (every 1 hour)")
    try:
        crawl_all_feeds()  # optional first run immediately
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down crawl scheduler")


if __name__ == "__main__":
    main()