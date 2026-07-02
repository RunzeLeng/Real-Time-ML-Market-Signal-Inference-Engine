import pandas as pd
import requests
from apify_client import ApifyClient
from src.common.config import ApifyConfig, ScrapeOpsConfig, config
from src.common.exceptions import RestartProcess
from src.processing.post_processing import PostProcessingService



class ApifyCrawlerService:
    
    def __init__(
        self,
        apify_config: ApifyConfig | None = None,
    ) -> None:
        self.config = apify_config or config.apify
        self.client = ApifyClient(self.config.token)
        self.post_processing_service = PostProcessingService()



    def crawl_default(self, num_posts: int = 5):
        try:
            run = self.client.actor(self.config.default_actor_id).call(
                run_input=self._build_default_run_input()
            )
            item = next(self.client.dataset(run["defaultDatasetId"]).iterate_items())

            df = self.post_processing_service.post_filtering(
                pd.DataFrame(item["posts"])[["id", "created_at", "content"]],
                num_posts=num_posts,
            )
            df = self.post_processing_service.post_formating(df, column="created_at")

        except Exception as error:
            print("Error during scraping:", error)

        return df



    def crawl_backup(self, num_posts: int = 5):
        try:
            run = self.client.actor(self.config.backup_actor_id).call(
                run_input=self._build_backup_run_input(num_posts=num_posts)
            )
            posts = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())

            content_only = [
                {
                    "id": post.get("id"),
                    "created_at": post.get("created_at"),
                    "content": post.get("content"),
                }
                for post in posts
            ]

            df = self.post_processing_service.post_filtering(
                pd.DataFrame(content_only, columns=["id", "created_at", "content"]),
                num_posts=num_posts,
            )
            df = self.post_processing_service.post_formating(df, column="created_at")

        except Exception as error:
            print("Error during scraping:", error)

        return df



    def _build_default_run_input(self) -> dict:
        return {
            "identifiers": [
                "https://truthsocial.com/@realDonaldTrump",
                "@realDonaldTrump",
                "realDonaldTrump",
            ],
            "fetchPosts": True,
        }
    
    
    
    def _build_backup_run_input(self, num_posts: int) -> dict:
        return {
            "username": "realDonaldTrump",
            "maxPosts": num_posts,
            "useLastPostId": False,
            "onlyReplies": False,
            "onlyMedia": False,
            "cleanContent": True,
        }



class CustomCrawlerService:
    
    def __init__(
        self,
        scrapeops_config: ScrapeOpsConfig | None = None,
    ) -> None:
        self.config = scrapeops_config or config.scrapeops
        self.session = requests.Session()
        self.post_processing_service = PostProcessingService()



    def crawl(
        self,
        num_posts: int = 5,
        extract_media: bool = False,
        apply_multimodal_filter: bool = False,
    ) -> pd.DataFrame:
        
        url = self.build_request_url()
        headers = self.build_headers()

        response = self.fetch_posts(url=url, headers=headers)
        df = self.extract_posts(response, extract_media=extract_media)

        if apply_multimodal_filter:
            df = self.post_processing_service.post_filtering_for_multimodal(df, num_posts=num_posts)
        else:
            df = self.post_processing_service.post_filtering(df, num_posts=num_posts)

        df = self.post_processing_service.post_formating(df, column="created_at")
        df.sort_values(by="created_at", ascending=False, inplace=True)

        return df
    
    
    
    def build_request_url(self) -> str:
        if not self.config.base_url:
            raise ValueError("Missing SCRAPEOPS_BASE_URL environment variable")

        params = {
            "exclude_replies": "true",
            "only_replies": "false",
            "with_muted": "true",
            "limit": "20",
        }

        return f"{self.config.base_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"



    def build_headers(self) -> dict:
        return {
            "accept": "application/json, text/plain, */*",
            "referer": "https://truthsocial.com/@realDonaldTrump",
        }



    def fetch_posts(
        self,
        url: str,
        headers: dict | None = None,
    ) -> list[dict]:
        
        if not self.config.api_key:
            raise ValueError("Missing SCRAPEOPS_API_KEY environment variable")

        if not self.config.endpoint:
            raise ValueError("Missing SCRAPEOPS_ENDPOINT environment variable")

        if headers:
            self.session.headers.update(headers)

        proxy_params = {
            "api_key": self.config.api_key,
            "url": url,
        }

        response = self.session.get(
            self.config.endpoint,
            params=proxy_params,
            timeout=120,
        )

        try:
            response.raise_for_status()
            return response.json()

        except requests.HTTPError as error:
            raise RestartProcess(f"ScrapeOps HTTP error: {error}")

        except ValueError as error:
            raise RestartProcess(f"Error parsing JSON response: {error}")



    def extract_posts(
        self,
        json_response: list[dict],
        extract_media: bool = False,
    ) -> pd.DataFrame:
        rows = []

        for post in json_response:
            rows.append({
                "id": post.get("id"),
                "created_at": post.get("created_at"),
                "content": self.fix_unicode(post.get("content", "")).strip(),
                "media": [
                    {
                        "url": media.get("url", ""),
                        "type": media.get("type", ""),
                    }
                    for media in post.get("media_attachments", [])
                ],
            })

        if extract_media:
            return pd.DataFrame(rows, columns=["id", "created_at", "content", "media"])

        return pd.DataFrame(rows, columns=["id", "created_at", "content"])
    
    
    
    def fix_unicode(self, text: str) -> str:
        try:
            return text.encode("utf-8").decode("unicode_escape")
        except Exception:
            return text
