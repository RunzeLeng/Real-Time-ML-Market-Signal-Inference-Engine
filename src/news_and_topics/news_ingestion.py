from datetime import datetime, timedelta
import time
import random
from zoneinfo import ZoneInfo
import pandas as pd
import requests
import trafilatura
from src.common.config import NewsConfig, config
from src.infrastructure.aws_s3 import S3StorageService
from src.prompt.news_search_string import news_search_string


class NewsIngestionService:
    
    def __init__(
        self,
        news_config: NewsConfig | None = None,
        s3_service: S3StorageService | None = None,
    ) -> None:
        self.config = news_config or config.news
        self.s3_service = s3_service or S3StorageService()



    def fetch_gdelt(
        self,
        start_et: datetime,
        end_et: datetime,
        chunk_in_hours: int = 6,
        domains: list[str] | None = None,
    ) -> pd.DataFrame:
        
        if domains is None:
            domains = ["yahoo.com", "us.cnn.com", "nbcnews.com"]

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
                response = requests.get(self.config.gdelt_base_url, params=params, timeout=60)

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

        return pd.DataFrame(all_articles)



    def fetch_news_api(
        self,
        search: str,
        search_fields: str | None = None,
        limit: int | None = None,
        max_pages: int | None = None,
        domains: list[str] | None = None,
        categories: str | None = None,
        exclude_categories: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
    ) -> pd.DataFrame:
        
        if not self.config.news_api_token:
            raise ValueError("Missing News API token.")

        search_fields = search_fields or self.config.news_api_search_fields
        limit = limit or self.config.news_api_limit
        max_pages = max_pages or self.config.news_api_max_pages
        categories = categories or self.config.news_api_categories
        exclude_categories = exclude_categories or self.config.news_api_exclude_categories

        all_articles = []

        for page in range(1, max_pages + 1):
            params = {
                "api_token": self.config.news_api_token,
                "search": search,
                "search_fields": search_fields,
                "language": self.config.news_api_language,
                "locale": self.config.news_api_locale,
                "limit": limit,
                "page": page,
                "categories": categories,
                "exclude_categories": exclude_categories,
                "published_after": published_after,
                "published_before": published_before,
            }

            if domains:
                params["domains"] = ",".join(domains) if len(domains) > 1 else domains[0]

            max_attempts = 5
            retryable_status_codes = {429, 500, 502, 503, 504, 525}

            for attempt in range(max_attempts):
                try:
                    response = requests.get(self.config.news_api_base_url, params=params, timeout=60)
                    response.raise_for_status()
                    
                    data = response.json()
                    break

                except requests.exceptions.JSONDecodeError:
                    print("TheNewsAPI did not return JSON.")
                    print("Status code:", response.status_code)
                    print("Response preview:", response.text[:500])
                    raise

                except (
                    requests.exceptions.HTTPError,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                ) as error:
                    
                    if isinstance(error, requests.exceptions.HTTPError):
                        status_code = error.response.status_code if error.response is not None else None
                        error_message = f"TheNewsAPI returned {status_code}"
                        
                        if status_code not in retryable_status_codes:
                            raise
                        
                    else:
                        error_message = f"TheNewsAPI request failed: {error}"

                    if attempt == max_attempts - 1:
                        raise

                    sleep_seconds = (2 ** attempt) + random.uniform(0, 1)
                    print(f"{error_message}. Retrying in {sleep_seconds:.2f} seconds...")
                    time.sleep(sleep_seconds)

            articles = data.get("data", [])
            all_articles.extend(articles)

            meta = data.get("meta", {})
            returned = meta.get("returned", len(articles))

            if returned < limit:
                break

        df = pd.DataFrame(all_articles)
        
        if df.empty or "url" not in df.columns:
            return pd.DataFrame()

        df["full_text"] = df["url"].apply(self.extract_full_article_text)
        df["full_text"] = df["full_text"].apply(self.clean_article_text)
        df = df[df["full_text"].fillna("").str.len() > 100].reset_index(drop=True)

        return df



    def clean_article_text(self, text: str) -> str:
        
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



    def extract_full_article_text(self, url: str) -> str | None:
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



    def fetch_by_date_windows(
        self,
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

                df_window = self.fetch_news_api(
                    search=news_search_string,
                    search_fields=self.config.news_api_search_fields,
                    limit=self.config.news_api_limit,
                    max_pages=self.config.news_api_max_pages,
                    categories=self.config.news_api_categories,
                    exclude_categories=self.config.news_api_exclude_categories,
                    published_after=published_after,
                    published_before=published_before,
                )

                if df_window is not None and not df_window.empty:
                    df_window = df_window.copy()
                    all_dfs.append(df_window)

            current_day += timedelta(days=1)

        return pd.concat(all_dfs, ignore_index=True)



    def update_daily_news(
        self,
        start_date: str,
        end_date: str,
        time_windows: list[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        
        df = self.fetch_by_date_windows(
            start_date=start_date,
            end_date=end_date,
            time_windows=time_windows,
        )

        self.s3_service.dedupe_and_save_news_by_date(df=df)

        return df


