import os
import logging
from src.common.config import setup_logging
from src.news_and_topics.news_ingestion import NewsIngestionService

logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("trafilatura").setLevel(logging.CRITICAL)



if __name__ == "__main__":
    
    setup_logging()
    
    start_date = os.getenv("NEWS_START_DATE", "2025-08-01")
    end_date = os.getenv("NEWS_END_DATE", "2025-08-31")
    
    logger.info(f"Fetching news from {start_date} to {end_date}")

    news_service = NewsIngestionService()
    df = news_service.update_daily_news(start_date=start_date, end_date=end_date)