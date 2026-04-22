from crawler import customized_crawler
from news_ingestion_pipeline import extract_full_article_text



if __name__ == "__main__":
    df = customized_crawler(num_posts=20, extract_media=True, apply_multimodal_filter=True)
    print(df)
