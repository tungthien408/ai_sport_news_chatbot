from crawler import news_crawler_v2
from news_pipeline.cleaner import TextCleaner
import json

def main():
    url: str = "https://vnexpress.net/rss/the-thao.rss"
    output_raw: str = '../logs/raw_output.json'
    output_clean: str = '../logs/cleaned_output.json' 
    news_crawler_v2.operation(url, output_raw)
    news_data: list[dict] = TextCleaner(output_raw).operation()

    with open(output_clean, "w", encoding="utf-8") as f:
        json.dump(news_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()