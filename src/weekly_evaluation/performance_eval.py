import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from aws_dynamodb import load_dynamodb_table_by_date_range
from aws_sns import publish_weekly_performance_to_sns
from crawler import duplicate_posts_to_minute_boundaries
from etf_historical import get_stock_bars, build_etf_vwap_future_changes


def get_today_and_last_monday_date_strings() -> tuple[str, str]:
    today = datetime.now()
    last_monday = today - timedelta(days=7)

    return today.strftime("%Y-%m-%d"), last_monday.strftime("%Y-%m-%d")



def is_today_monday() -> bool:
    return datetime.now().weekday() == 0



def build_prediction_performance_summary(
    df: pd.DataFrame,
    metric_1_col: str = "vwap_pct_change_30m",
    # metric_2_col: str = "vwap_pct_change_3h",
    reasonableness_col: str = "reasonableness_score",
    high_reasonableness_threshold: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working_df = df.loc[
        df[metric_1_col].notna() 
        # & df[metric_2_col].notna()
    ].copy()

    metric_1_name = metric_1_col.rsplit("_", 1)[-1]
    # metric_2_name = metric_2_col.rsplit("_", 1)[-1]

    for metric_col, metric_name in [
        (metric_1_col, metric_1_name),
        # (metric_2_col, metric_2_name),
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
            f"accuracy_{metric_1_name}": sub_df[f"binary_{metric_1_name}"].mean(),
            # f"accuracy_{metric_2_name}": sub_df[f"binary_{metric_2_name}"].mean(),
            f"avg_{metric_1_name}_return": sub_df[f"aligned_{metric_1_name}_return"].mean(),
            # f"avg_{metric_2_name}_return": sub_df[f"aligned_{metric_2_name}_return"].mean(),
            f"potential_total_{metric_1_name}_return": len(sub_df) * sub_df[f"aligned_{metric_1_name}_return"].mean(),
            # f"potential_total_{metric_2_name}_return": len(sub_df) * sub_df[f"aligned_{metric_2_name}_return"].mean(),
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



def weekly_performance_review() -> None:
    if not is_today_monday():
        return
    
    try:
        print("Starting weekly performance review...")
        today_date, last_monday_date = get_today_and_last_monday_date_strings()

        etf_df = get_stock_bars(last_monday_date, today_date)
        etf_df = build_etf_vwap_future_changes(etf_df)
        
        published_df = load_dynamodb_table_by_date_range("published_records", start_date=last_monday_date, end_date=today_date)
        
        if published_df.empty:
            print("No published records found for the past week.")
            return
        else:
            print(f"Found {len(published_df)} published records for the past week.")
            
            published_df = duplicate_posts_to_minute_boundaries(
                df=published_df,
                datetime_column="created_at",
                post_duplicate=False,
            )
            
            weekly_evaluation_df = published_df.merge(
                etf_df,
                left_on=["symbol", "created_at"],
                right_on=["symbol", "timestamp"],
                how="inner",
            )

            overall_summary, symbol_level_summary, buy_level_summary, sell_level_summary, reasonableness_level_summary \
            = build_prediction_performance_summary(weekly_evaluation_df)
            
            publish_weekly_performance_to_sns(
                overall_df=overall_summary,
                symbol_level_df=symbol_level_summary,
                buy_level_df=buy_level_summary,
                sell_level_df=sell_level_summary,
                reasonableness_level_df=reasonableness_level_summary,
                topic_arn=os.getenv("AWS_SNS_TOPIC_ARN"),
            )
            
            print("Weekly performance review published to SNS.")
     
    except Exception as e:
        print(f"Weekly performance review failed: {e}")



if __name__ == "__main__":
    
    weekly_performance_review()
    print("Weekly performance review completed.")