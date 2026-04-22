import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import os
from dotenv import load_dotenv
from aws_s3 import load_parquet_from_s3, save_df_to_s3_parquet
from crawler import filter_posts_by_date_and_content_length, duplicate_posts_to_minute_boundaries, add_post_prefix_to_content


def get_stock_bars(start_date: str = "2025-01-01", end_date: str = "2026-03-25") -> pd.DataFrame:
    
    load_dotenv()
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    
    etf_list = [
        "QQQ",   # Nasdaq-100 exposure
        "SPY",   # S&P 500 exposure
        "DIA",   # Dow Jones Industrial Average exposure
        "UCO",   # Leveraged long crude oil exposure
        "TLT",   # Long-duration U.S. Treasury bond exposure
        "UGL",   # Leveraged long gold exposure
        "VXX",   # Short-term volatility exposure
        "TSLL",  # Leveraged Tesla exposure
        "NVDU",  # Leveraged Nvidia exposure
        "SOXX",  # Semiconductor industry exposure
        "XLF",   # U.S. financial sector exposure
        "XLE",   # U.S. energy sector exposure
        "HYG",   # High-yield corporate bond exposure
    ]
    
    request_params = StockBarsRequest(
                            symbol_or_symbols=etf_list,
                            timeframe=TimeFrame.Minute,
                            start=datetime.strptime(start_date, '%Y-%m-%d'),
                            end=datetime.strptime(end_date, '%Y-%m-%d')
                            )

    bars = client.get_stock_bars(request_params)
    
    df = bars.df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("US/Eastern").dt.floor("s")
    df = df.sort_values(["symbol", "timestamp"]).copy()

    # convert to dataframe
    return df


  
def build_etf_vwap_future_changes(df: pd.DataFrame) -> pd.DataFrame:
    etf_df = df.copy()
    etf_df["timestamp"] = pd.to_datetime(etf_df["timestamp"], errors="coerce")
    etf_df = etf_df.sort_values(["symbol", "timestamp"]).copy()

    result = etf_df[["symbol", "timestamp", "vwap"]].copy()

    horizons = {
        "vwap_pct_change_5m": pd.Timedelta(minutes=5),
        "vwap_pct_change_10m": pd.Timedelta(minutes=10),
        "vwap_pct_change_30m": pd.Timedelta(minutes=30),
        "vwap_pct_change_1h": pd.Timedelta(hours=1),
        "vwap_pct_change_3h": pd.Timedelta(hours=3),
    }

    lookup = (
        etf_df[["symbol", "timestamp", "vwap"]]
        .set_index(["symbol", "timestamp"])["vwap"]
    )

    for new_col, delta in horizons.items():
        future_index = pd.MultiIndex.from_frame(
            result.assign(timestamp=result["timestamp"] + delta)[["symbol", "timestamp"]]
        )

        future_vwap = lookup.reindex(future_index).to_numpy()
        current_vwap = result["vwap"].to_numpy()

        result[new_col] = ((future_vwap - current_vwap) / current_vwap) * 100

    return result[
        [
            "symbol",
            "timestamp",
            "vwap",
            "vwap_pct_change_5m",
            "vwap_pct_change_10m",
            "vwap_pct_change_30m",
            "vwap_pct_change_1h",
            "vwap_pct_change_3h",
        ]
    ]



def post_etf_joins(
    bucket_name: str,
    post_object_key: str,
    etf_object_key: str,
    post_start_date: str,
    post_end_date: str,
    min_content_length: int = 50,
    post_duplicate: bool = False,
) -> pd.DataFrame:
    
    post_df = load_parquet_from_s3(
        bucket_name=bucket_name,
        object_key=post_object_key,
        num_posts=100000
    )

    etf_df = load_parquet_from_s3(
        bucket_name=bucket_name,
        object_key=etf_object_key,
    )

    post_df = filter_posts_by_date_and_content_length(
        df=post_df,
        start_date=post_start_date,
        end_date=post_end_date,
        min_content_length=min_content_length,
        date_column="created_at",
        content_column="content",
    )
    
    post_df = duplicate_posts_to_minute_boundaries(
        df=post_df,
        datetime_column="created_at",
        post_duplicate=post_duplicate,
    )
    
    post_df = add_post_prefix_to_content(post_df)
    
    etf_df = build_etf_vwap_future_changes(etf_df)

    joined_df = post_df.merge(
        etf_df,
        how="inner",
        left_on="created_at",
        right_on="timestamp",
    )

    return joined_df



def extract_ids_and_contents(df):
    ids = df["id"].astype(str).tolist()
    contents = df["content"].astype(str).tolist()
    return ids, contents



def deduplicate_and_remove_existing_ids(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.DataFrame:
    df1_dedup = df1.drop_duplicates(subset=["id"]).copy()
    existing_ids = set(df2["id"].dropna())

    result_df = df1_dedup[~df1_dedup["id"].isin(existing_ids)].copy()
    return result_df



if __name__ == "__main__":
    load_dotenv()
    # df = post_etf_joins(
    #     bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
    #     post_object_key=os.getenv("AWS_S3_OBJECT_KEY_POST"),
    #     etf_object_key=os.getenv("AWS_S3_OBJECT_KEY_ETF"),
    #     post_start_date="2025-01-01",
    #     post_end_date="2025-12-31",
    #     min_content_length=50,
    #     post_duplicate=False,
    # )
    # print(df)
    # df = get_stock_bars()
    
    # print(df.head(50))
    # print(df.shape)
    
    df= load_parquet_from_s3(
        bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
        object_key=os.getenv("AWS_S3_OBJECT_KEY_ETF"),
    )
    print(df)
    
    print(df.groupby("symbol").size().sort_values(ascending=False))
    
    # df = build_etf_vwap_future_changes(df)
    # print(df)
    
    
    # df = get_stock_bars()
    # print(df)
    
    # save_df_to_s3_parquet(df, bucket_name=os.getenv("AWS_S3_BUCKET_NAME"), object_key=os.getenv("AWS_S3_OBJECT_KEY_ETF"))