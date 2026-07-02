from io import BytesIO
from pathlib import PurePosixPath
import boto3
import pandas as pd
from src.common.config import config
from src.processing.post_processing import PostProcessingService



class S3StorageService:
    
    def __init__(self):
        self.config = config.s3
        self.client = boto3.client(
            "s3",
            region_name=self.config.region,
        )
        self.post_processing_service = PostProcessingService()



    def read_parquet(
        self,
        object_key: str,
        num_posts: int = 10000,
        apply_project_formatting: bool = True,
    ) -> pd.DataFrame:
        
        response = self.client.get_object(
            Bucket=self.config.bucket_name,
            Key=object_key,
        )

        df = pd.read_parquet(BytesIO(response["Body"].read()))

        if apply_project_formatting:
            df = self._apply_post_or_etf_formatting(
                df=df,
                object_key=object_key,
                num_posts=num_posts,
            )

        print(df)
        return df



    def read_group_parquet(
        self,
        object_key_keyword: str,
        num_posts: int = 10000,
        apply_project_formatting: bool = True,
    ) -> pd.DataFrame:
        
        response = self.client.list_objects_v2(
            Bucket=self.config.bucket_name,
        )
        matching_keys = []

        for obj in response.get("Contents", []):
            key = obj["Key"]
            file_name = PurePosixPath(key).name

            if key.lower().endswith(".parquet") and object_key_keyword.lower() in file_name.lower():
                matching_keys.append(key)

        if not matching_keys:
            raise ValueError(f"No parquet files found with keyword: {object_key_keyword}")

        df_list = []

        for key in matching_keys:
            response = self.client.get_object(
                Bucket=self.config.bucket_name,
                Key=key,
            )
            df_part = pd.read_parquet(BytesIO(response["Body"].read()))
            df_list.append(df_part)

        df = pd.concat(df_list, ignore_index=True)

        if apply_project_formatting:
            df = self._apply_post_or_etf_formatting(
                df=df,
                object_key=object_key_keyword,
                num_posts=num_posts,
            )

        print(df)
        return df



    def save_df_as_parquet(
        self,
        df: pd.DataFrame,
        object_key: str,
    ) -> None:
        
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        self.client.put_object(
            Bucket=self.config.bucket_name,
            Key=object_key,
            Body=buffer.getvalue(),
        )



    def dedupe_and_save_news_by_date(
        self,
        df: pd.DataFrame,
        object_key: str | None = None,
    ) -> None:
        
        object_key = object_key or self.config.daily_news_object_key

        if not object_key:
            raise ValueError("Missing daily news S3 object key.")

        df = df.copy()
        df = df.drop_duplicates(subset=["uuid"]).reset_index(drop=True)
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
        df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
        df = df[df["full_text"].fillna("").astype(str).str.len() >= 200].reset_index(drop=True)

        df["published_at"] = pd.to_datetime(
            df["published_at"],
            utc=True,
            errors="coerce",
        )
        df = df[df["published_at"].notna()].reset_index(drop=True)

        for published_date, date_df in df.groupby(df["published_at"].dt.date):
            date_string = published_date.strftime("%Y-%m-%d")
            s3_key = f"{object_key}/{date_string}/news.parquet"

            self.save_df_as_parquet(
                df=date_df,
                object_key=s3_key,
            )

            print(f"{len(date_df)} Daily news saved to s3://{self.config.bucket_name}/{s3_key}")



    def read_parquet_prefix(
        self,
        prefix: str,
    ) -> pd.DataFrame:
        
        prefix = prefix.strip("/")
        all_dfs = []
        continuation_token = None

        while True:
            list_kwargs = {
                "Bucket": self.config.bucket_name,
                "Prefix": prefix,
            }

            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            response = self.client.list_objects_v2(**list_kwargs)

            for item in response.get("Contents", []):
                key = item["Key"]

                object_response = self.client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=key,
                )

                parquet_bytes = object_response["Body"].read()
                df_file = pd.read_parquet(BytesIO(parquet_bytes))
                all_dfs.append(df_file)

            if not response.get("IsTruncated"):
                break

            continuation_token = response.get("NextContinuationToken")

        return pd.concat(all_dfs, ignore_index=True)



    def read_daily_news(self) -> pd.DataFrame:
        if not self.config.daily_news_object_key:
            raise ValueError("Missing daily news S3 object key.")

        return self.read_parquet_prefix(
            prefix=self.config.daily_news_object_key,
        )



    def read_post_data(self, num_posts: int = 10000) -> pd.DataFrame:
        if not self.config.post_object_key:
            raise ValueError("Missing post S3 object key.")

        return self.read_parquet(
            object_key=self.config.post_object_key,
            num_posts=num_posts,
        )



    def read_etf_data(self, num_posts: int = 10000) -> pd.DataFrame:
        if not self.config.etf_object_key:
            raise ValueError("Missing ETF S3 object key.")

        return self.read_parquet(
            object_key=self.config.etf_object_key,
            num_posts=num_posts,
        )



    def _apply_post_or_etf_formatting(
        self,
        df: pd.DataFrame,
        object_key: str,
        num_posts: int,
    ) -> pd.DataFrame:
        
        if "etf" not in object_key.lower():
            df = self.post_processing_service.post_filtering(df[["id", "created_at", "content"]].copy(), num_posts=num_posts)
            df = self.post_processing_service.post_formating(df, column="created_at")
            
        else:
            from src.market_data.etf_market_data import EtfMarketDataService
            df = EtfMarketDataService.format_etf_data(df, column="timestamp")

        return df
