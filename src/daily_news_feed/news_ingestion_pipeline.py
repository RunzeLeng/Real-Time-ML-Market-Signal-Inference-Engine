from datetime import datetime, timedelta
import os
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import requests
import pandas as pd
from aws_s3 import dedupe_and_save_news_to_s3_by_date, read_parquet_files_from_s3_prefix
from news_domains import news_domains
from prompt.news_search_string import news_search_string
import time
import trafilatura


def fetch_news_gdelt(
    start_et: datetime,
    end_et: datetime,
    chunk_in_hours: int = 6,
    domains: list[str] = ["yahoo.com", "us.cnn.com", "nbcnews.com"],
) -> pd.DataFrame:
    
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    query = (
        "(market OR stocks OR finance OR economy OR etf OR geopolitics OR iran OR israel) "
        "sourcelang:english sourcecountry:US"
    )

    if domains:
        if len(domains) == 1:
            domain_query = f"domainis:{domains[0]}"
        else:
            domain_query = " OR ".join(f"domainis:{domain}" for domain in domains)
            domain_query = f"({domain_query})"

        query = f"{query} {domain_query}"

    all_articles = []
    current_start = start_et

    while current_start < end_et:
        current_end = min(current_start + timedelta(hours=chunk_in_hours), end_et)

        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "sort": "dateasc",
            "maxrecords": 250,
            "startdatetime": current_start.astimezone(ZoneInfo("UTC")).strftime("%Y%m%d%H%M%S"),
            "enddatetime": current_end.astimezone(ZoneInfo("UTC")).strftime("%Y%m%d%H%M%S"),
        }

        for attempt in range(6):
            response = requests.get(base_url, params=params, timeout=60)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = int(retry_after) if retry_after else 10 * (attempt + 1)
                print(f"GDELT rate limited request. Waiting {sleep_seconds} seconds...")
                time.sleep(sleep_seconds)
                continue

            response.raise_for_status()

            if not response.text.strip():
                articles = []
            else:
                try:
                    data = response.json()
                except requests.exceptions.JSONDecodeError:
                    print("GDELT did not return valid JSON.")
                    print("Status code:", response.status_code)
                    print("Content-Type:", response.headers.get("Content-Type"))
                    print("Response preview:", response.text[:500])
                    articles = []
                else:
                    articles = data.get("articles", [])

            if len(articles) == 250:
                print(
                    f"Warning: hit 250-record cap from {current_start} to {current_end}. "
                    "Use a smaller chunk_minutes value."
                )

            all_articles.extend(articles)
            break

        time.sleep(5)
        current_start = current_end

    df = pd.DataFrame(all_articles)
    
    return df



def fetch_news_thenewsapi(
    api_token: str,
    search: str,
    search_fields: str = "title,main_text",
    limit: int = 25,
    max_pages: int = 10,
    domains: list[str] | None = None,
    categories: str = "business,politics",
    exclude_categories: str = "travel,food",
    published_after: str | None = None,
    published_before: str | None = None,
) -> pd.DataFrame:

    if not api_token:
        raise ValueError("Missing TheNewsAPI token.")

    base_url = "https://api.thenewsapi.com/v1/news/all"

    all_articles = []

    for page in range(1, max_pages + 1):
        params = {
            "api_token": api_token,
            "search": search,
            "search_fields": search_fields,
            "language": "en",
            "locale": "us",
            "limit": limit,
            "page": page,
            "categories": categories,
            "exclude_categories": exclude_categories,
            "published_after": published_after,
            "published_before": published_before,
        }
        
        if domains:
            params["domains"] = ",".join(domains) if len(domains) > 1 else domains[0]
        
        response = requests.get(base_url, params=params, timeout=60)

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            print("TheNewsAPI did not return JSON.")
            print("Status code:", response.status_code)
            print("Response preview:", response.text[:500])
            response.raise_for_status()

        response.raise_for_status()

        articles = data.get("data", [])
        all_articles.extend(articles)

        meta = data.get("meta", {})
        returned = meta.get("returned", len(articles))

        if returned < limit:
            break

    df = pd.DataFrame(all_articles)
    
    df["full_text"] = df["url"].apply(extract_full_article_text)
    df["full_text"] = df["full_text"].apply(clean_article_text)
    df = df[df["full_text"].fillna("").str.len() > 100].reset_index(drop=True)

    return df



def clean_article_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    if not text.strip():
        return ""

    bad_phrases = [
        "sign up",
        "subscribe",
        "newsletter",
        "advertisement",
        "related article",
        "related articles",
        "read more",
        "follow us",
        "share this",
        "all rights reserved",
        "cookie",
        "privacy policy",
        "terms of service",
    ]

    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if len(line) < 30:
            continue

        lower_line = line.lower()

        if any(phrase in lower_line for phrase in bad_phrases):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)



def extract_full_article_text(url: str) -> str | None:

    try:
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return None

        return trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            include_images=False,
            include_links=False,
            favor_precision=True,
            deduplicate=True,
        )
    except Exception as error:
        print(f"Failed to extract article: {url}")
        print(error)
        return None



def fetch_news_by_date_windows(
    start_date: str,
    end_date: str,
    time_windows: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    
    eastern = ZoneInfo("America/New_York")
    utc = ZoneInfo("UTC")
    
    def build_et_datetime(day_string: str, time_string: str) -> datetime:
        return datetime.strptime(
            f"{day_string} {time_string}",
            "%Y-%m-%d %H:%M",
        ).replace(tzinfo=eastern)
    
    if time_windows is None:
        time_windows = [
            ("06:00", "10:00"),
            ("10:00", "14:00"),
            ("14:00", "18:00"),
        ]

    start_day = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_day = datetime.strptime(end_date, "%Y-%m-%d").date()

    all_dfs = []
    current_day = start_day

    while current_day <= end_day:
        print(f"Fetching news for {current_day}...")
        day_string = current_day.strftime("%Y-%m-%d")

        for start_time, end_time in time_windows:
            window_start_et = build_et_datetime(day_string, start_time)
            window_end_et = build_et_datetime(day_string, end_time)

            window_start_utc = window_start_et.astimezone(utc)
            window_end_utc = window_end_et.astimezone(utc)

            published_after = window_start_utc.strftime("%Y-%m-%dT%H:%M")
            published_before = window_end_utc.strftime("%Y-%m-%dT%H:%M")

            df_window = fetch_news_thenewsapi(
                api_token=os.getenv("THE_NEWS_API_TOKEN"),
                search=news_search_string,
                search_fields="title,main_text",
                limit=25,
                max_pages=20,
                categories="business,politics",
                exclude_categories="travel,food,entertainment,health,sports",
                published_after=published_after,
                published_before=published_before,
            )

            if df_window is not None and not df_window.empty:
                df_window = df_window.copy()
                all_dfs.append(df_window)

        current_day += timedelta(days=1)

    df = pd.concat(all_dfs, ignore_index=True)
    
    return df



if __name__ == "__main__":
    load_dotenv()
    
    # df = fetch_news_gdelt(
    #     start_et=datetime(2026, 4, 8, 0, 0, 0, tzinfo=ZoneInfo("America/New_York")),
    #     end_et=datetime(2026, 4, 12, 0, 0, 0, tzinfo=ZoneInfo("America/New_York")),
    #     chunk_in_hours=96,
    # )

    # df = fetch_news_thenewsapi(
    #     api_token=os.getenv("THE_NEWS_API_TOKEN"),
    #     search=news_search_string,
    #     search_fields="title,main_text",
    #     limit=25,
    #     max_pages=20,
    #     categories="business,politics",
    #     exclude_categories="travel,food,entertainment,health,sports",
    #     published_after="2026-04-09",
    #     published_before="2026-04-12",
    # )
    
    df = fetch_news_by_date_windows(
        start_date="2026-04-13",
        end_date="2026-04-15",
    )
    
    dedupe_and_save_news_to_s3_by_date(
        df, 
        bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
        object_key=os.getenv("AWS_S3_OBJECT_KEY_DAILY_NEWS")
    )
    
    df = read_parquet_files_from_s3_prefix(
            bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
            prefix=os.getenv("AWS_S3_OBJECT_KEY_DAILY_NEWS"),
    )
    
    print(df)