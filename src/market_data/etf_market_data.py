from datetime import datetime
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from src.common.config import MarketDataConfig, config
from src.common.etf_constants import ETF_LIST
from src.infrastructure.aws_s3 import S3StorageService
from src.processing.post_processing import PostProcessingService


class EtfMarketDataService:

    def __init__(
        self,
        market_data_config: MarketDataConfig | None = None,
        s3_service: S3StorageService | None = None,
    ) -> None:

        self.config = market_data_config or config.market_data
        self.s3_service = s3_service or S3StorageService()
        self.post_processing_service = PostProcessingService()
        self.client = StockHistoricalDataClient(
            self.config.alpaca_api_key,
            self.config.alpaca_secret_key,
        )



    def get_stock_bars(
        self,
        start_date: str = "2025-01-01",
        end_date: str = "2026-03-25",
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:

        etf_list = symbols or ETF_LIST

        request_params = StockBarsRequest(
            symbol_or_symbols=etf_list,
            timeframe=TimeFrame.Minute,
            start=datetime.strptime(start_date, "%Y-%m-%d"),
            end=datetime.strptime(end_date, "%Y-%m-%d"),
        )

        bars = self.client.get_stock_bars(request_params)

        df = bars.df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("US/Eastern").dt.floor("s")
        df = df.sort_values(["symbol", "timestamp"]).copy()

        return df



    def build_etf_vwap_future_changes(self, df: pd.DataFrame) -> pd.DataFrame:
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



    def join_posts_with_etf_features(
        self,
        post_start_date: str,
        post_end_date: str,
        min_content_length: int = 50,
        post_duplicate: bool = False,
    ) -> pd.DataFrame:

        post_df = self.s3_service.read_post_data(num_posts=100000)

        etf_df = self.s3_service.read_etf_data()

        post_df = self.post_processing_service.filter_posts_by_date_and_content_length(
            df=post_df,
            start_date=post_start_date,
            end_date=post_end_date,
            min_content_length=min_content_length,
            date_column="created_at",
            content_column="content",
        )

        post_df = self.post_processing_service.duplicate_posts_to_minute_boundaries(
            df=post_df,
            datetime_column="created_at",
            post_duplicate=post_duplicate,
        )

        post_df = self.post_processing_service.add_post_prefix_to_content(post_df)

        etf_df = self.build_etf_vwap_future_changes(etf_df)

        joined_df = post_df.merge(
            etf_df,
            how="inner",
            left_on="created_at",
            right_on="timestamp",
        )

        return joined_df



    @staticmethod
    def format_etf_data(
        df: pd.DataFrame,
        column: str = "timestamp",
    ) -> pd.DataFrame:
        
        formatted_df = df.copy()
        formatted_df[column] = pd.to_datetime(formatted_df[column], errors="coerce")
        formatted_df[column] = formatted_df[column].dt.floor("s").dt.tz_convert("US/Eastern")

        return formatted_df
