from datetime import datetime, timedelta
import pandas as pd
from src.common.config import PerformanceReviewConfig, config
from src.infrastructure.aws_aurora_dsql import AuroraDsqlClient
from src.infrastructure.aws_dynamodb import DynamoDBService
from src.infrastructure.aws_sns import SnsNotificationService
from src.market_data.etf_market_data import EtfMarketDataService
from src.processing.post_processing import PostProcessingService



class PerformanceReviewService:
    
    def __init__(
        self,
        performance_review_config: PerformanceReviewConfig | None = None,
        dynamodb_service: DynamoDBService | None = None,
        aurora_client: AuroraDsqlClient | None = None,
        sns_service: SnsNotificationService | None = None,
        etf_market_data_service: EtfMarketDataService | None = None,
        post_processing_service: PostProcessingService | None = None,
    ) -> None:
        
        self.config = performance_review_config or config.performance_review
        self.dynamodb_service = dynamodb_service or DynamoDBService()
        self.aurora_client = aurora_client or AuroraDsqlClient()
        self.sns_service = sns_service or SnsNotificationService()
        self.etf_market_data_service = etf_market_data_service or EtfMarketDataService()
        self.post_processing_service = post_processing_service or PostProcessingService()



    def merge_overall_model_accuracy(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        
        sql_query = """
        SELECT symbol, AVG(avg_valid_accuracy_high_confidence) AS model_accuracy
        FROM training_output.selected_model_performance
        GROUP BY symbol
        """

        rows = self.aurora_client.dsql_execute_sql(sql_query)
        merged_df = merged_df.merge(rows, on="symbol", how="inner")

        return merged_df



    def get_review_date_range(self) -> tuple[str, str]:
        today = datetime.now()
        start_date = today - timedelta(days=self.config.lookback_days)
        
        return today.strftime("%Y-%m-%d"), start_date.strftime("%Y-%m-%d")



    def is_review_day(self) -> bool:
        return datetime.now().weekday() == self.config.review_weekday



    def build_prediction_performance_summary(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        default_metric_col = self.config.default_metric_col
        # secondary_metric_col = self.config.secondary_metric_col
        
        reasonableness_col = self.config.reasonableness_col
        high_reasonableness_threshold = self.config.high_reasonableness_threshold

        working_df = df.loc[
            df[default_metric_col].notna()
            # & df[secondary_metric_col].notna()
        ].copy()

        default_metric_name = default_metric_col.rsplit("_", 1)[-1]
        # secondary_metric_name = secondary_metric_col.rsplit("_", 1)[-1]

        for metric_col, metric_name in [
            (default_metric_col, default_metric_name),
            # (secondary_metric_col, secondary_metric_name),
        ]:
            working_df[f"actual_signal_{metric_name}"] = working_df[metric_col].apply(
                lambda x: "sell" if x < 0 else "buy"
            )

            working_df[f"binary_{metric_name}"] = (
                working_df["predicted_signal"] == working_df[f"actual_signal_{metric_name}"]
            ).astype(int)

            working_df[f"aligned_{metric_name}_return"] = working_df.apply(
                lambda row: row[metric_col] if row["predicted_signal"] == "buy" else -row[metric_col],
                axis=1,
            )


        def summarize(sub_df: pd.DataFrame) -> pd.Series:
            
            return pd.Series({
                "total_signals": len(sub_df),
                f"accuracy_{default_metric_name}": sub_df[f"binary_{default_metric_name}"].mean(),
                # f"accuracy_{secondary_metric_name}": sub_df[f"binary_{secondary_metric_name}"].mean(),
                f"avg_{default_metric_name}_return": sub_df[f"aligned_{default_metric_name}_return"].mean(),
                # f"avg_{secondary_metric_name}_return": sub_df[f"aligned_{secondary_metric_name}_return"].mean(),
                f"potential_total_{default_metric_name}_return": (
                    len(sub_df) * sub_df[f"aligned_{default_metric_name}_return"].mean()
                ),
                # f"potential_total_{secondary_metric_name}_return": (
                #     len(sub_df) * sub_df[f"aligned_{secondary_metric_name}_return"].mean()
                # ),
            })

        summary_dict = {
            "overall": summarize(working_df).to_frame().T,
            "symbol_level": working_df.groupby("symbol").apply(summarize).reset_index(),
            "buy_level": summarize(working_df.loc[working_df["predicted_signal"] == "buy"]).to_frame().T,
            "sell_level": summarize(working_df.loc[working_df["predicted_signal"] == "sell"]).to_frame().T,
            "reasonableness_level": summarize(
                working_df.loc[working_df[reasonableness_col] >= high_reasonableness_threshold - 1e-9]
            ).to_frame().T,
        }

        return (
            summary_dict["overall"],
            summary_dict["symbol_level"],
            summary_dict["buy_level"],
            summary_dict["sell_level"],
            summary_dict["reasonableness_level"],
        )



    def run_review(self) -> None:
        
        if not self.is_review_day():
            return

        try:
            print("Starting performance review...")
            today_date, start_date = self.get_review_date_range()

            etf_df = self.etf_market_data_service.get_stock_bars(start_date, today_date)
            etf_df = self.etf_market_data_service.build_etf_vwap_future_changes(etf_df)

            published_df = self.dynamodb_service.load_published_records_by_date_range(
                start_date=start_date,
                end_date=today_date,
            )

            if published_df.empty:
                print("No published records found for the review period.")
                return
            
            else:
                print(f"Found {len(published_df)} published records for the review period.")

                published_df = self.post_processing_service.duplicate_posts_to_minute_boundaries(
                    df=published_df,
                    datetime_column="created_at",
                    post_duplicate=False,
                )

                evaluation_df = published_df.merge(
                    etf_df,
                    left_on=["symbol", "created_at"],
                    right_on=["symbol", "timestamp"],
                    how="inner",
                )

                overall_summary, symbol_level_summary, buy_level_summary, sell_level_summary, reasonableness_level_summary \
                    = self.build_prediction_performance_summary(evaluation_df)

                self.sns_service.publish_weekly_performance(
                    overall_df=overall_summary,
                    symbol_level_df=symbol_level_summary,
                    buy_level_df=buy_level_summary,
                    sell_level_df=sell_level_summary,
                    reasonableness_level_df=reasonableness_level_summary,
                )

                print("Performance review published to SNS.")

        except Exception as error:
            print(f"Performance review failed: {error}")
