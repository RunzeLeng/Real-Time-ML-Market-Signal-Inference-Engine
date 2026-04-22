from io import BytesIO
import os
from pathlib import PurePosixPath
import boto3
from dotenv import load_dotenv
import pandas as pd
from crawler import post_filtering, post_formating, etf_formating


def load_parquet_from_s3(
    bucket_name: str,
    object_key: str,
    num_posts: int = 10000
) -> pd.DataFrame:
    """
    Read a parquet file from S3 and keep only the ID, created_by, and content columns.
    """
    
    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)

    df = pd.read_parquet(BytesIO(response["Body"].read()))
    
    if "etf" not in object_key.lower():
        df = post_filtering(df[["id", "created_at", "content"]].copy(), num_posts=num_posts)
        df = post_formating(df, column="created_at")
    else:
        df = etf_formating(df, column="timestamp")
    
    print(df)
    return df



def load_group_parquet_from_s3(
    bucket_name: str,
    object_key: str,
    num_posts: int = 10000,
) -> pd.DataFrame:
    s3_client = boto3.client("s3")

    response = s3_client.list_objects_v2(Bucket=bucket_name)
    matching_keys = []

    for obj in response.get("Contents", []):
        key = obj["Key"]
        file_name = PurePosixPath(key).name

        if key.lower().endswith(".parquet") and object_key.lower() in file_name.lower():
            matching_keys.append(key)

    if not matching_keys:
        raise ValueError(f"No parquet files found with keyword: {object_key}")

    df_list = []

    for key in matching_keys:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        df_part = pd.read_parquet(BytesIO(response["Body"].read()))
        df_list.append(df_part)

    df = pd.concat(df_list, ignore_index=True)

    if "etf" not in object_key.lower():
        df = post_filtering(df[["id", "created_at", "content"]].copy(), num_posts=num_posts)
        df = post_formating(df, column="created_at")
    else:
        df = etf_formating(df, column="timestamp")

    print(df)
    return df



def save_df_to_s3_parquet(df: pd.DataFrame, bucket_name: str, object_key: str) -> None:
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=buffer.getvalue()
    )



def dedupe_and_save_news_to_s3_by_date(
    df: pd.DataFrame,
    bucket_name: str,
    object_key: str,
):    
    df = df.copy()
    df = df.drop_duplicates(subset=["uuid"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    df = df[df["full_text"].fillna("").astype(str).str.len() >= 200].reset_index(drop=True)

    df["published_at"] = pd.to_datetime(
        df["published_at"],
        utc=True,
        errors="coerce",)
    df = df[df["published_at"].notna()].reset_index(drop=True)

    s3_client = boto3.client("s3")

    for published_date, date_df in df.groupby(df["published_at"].dt.date):
        date_string = published_date.strftime("%Y-%m-%d")

        s3_key = f"{object_key}/{date_string}/news.parquet"

        buffer = BytesIO()
        date_df.to_parquet(buffer, index=False)
        buffer.seek(0)

        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=buffer.getvalue(),
        )

        print(f"{len(date_df)} Daily news saved to s3://{bucket_name}/{s3_key}")



def read_parquet_files_from_s3_prefix(
    bucket_name: str,
    prefix: str,
) -> pd.DataFrame:
    
    s3_client = boto3.client("s3")
    prefix = prefix.strip("/")

    all_dfs = []
    continuation_token = None

    while True:
        list_kwargs = {
            "Bucket": bucket_name,
            "Prefix": prefix,
        }

        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**list_kwargs)

        for item in response.get("Contents", []):
            key = item["Key"]

            object_response = s3_client.get_object(
                Bucket=bucket_name,
                Key=key,
            )

            parquet_bytes = object_response["Body"].read()
            df_file = pd.read_parquet(BytesIO(parquet_bytes))
            all_dfs.append(df_file)

        if not response.get("IsTruncated"):
            break
        else:
            continuation_token = response.get("NextContinuationToken")
    
    return pd.concat(all_dfs, ignore_index=True)



if __name__ == "__main__":
    load_dotenv()
    load_parquet_from_s3(
        bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
        object_key=os.getenv("AWS_S3_OBJECT_KEY_ETF"),
    )