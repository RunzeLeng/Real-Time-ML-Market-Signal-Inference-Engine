import os
import boto3
import pandas as pd
from aws_aurora_dsql import dsql_execute_sql


def merge_overall_model_accuracy(merged_df: pd.DataFrame) -> pd.DataFrame:
    sql_query = """
    SELECT symbol, AVG(avg_valid_accuracy_high_confidence) AS model_accuracy
    FROM training_output.selected_model_performance
    GROUP BY symbol
    """
    
    rows = dsql_execute_sql(
        host=os.getenv("AWS_AURORA_DB_HOST"),
        database="postgres",
        sql=sql_query,
        user="admin",
        region="us-east-1",
        profile="default",
    )
    
    merged_df = merged_df.merge(
        rows, on="symbol", how="inner"
    )
    
    return merged_df



def publish_etf_signals_to_sns(
    merged_df: pd.DataFrame,
    topic_arn: str | None = None,
    subject: str = "ETF Market Signal Alert",
    region_name: str = "us-east-1",
) -> dict:
    sns_client = boto3.client("sns", region_name=region_name)

    message_lines = []

    message_lines.append(f"Post ID: {merged_df.iloc[0]['id']}")
    message_lines.append(f"Created At: {merged_df.iloc[0]['created_at']}")
    message_lines.append(f"Market Impact Score: {merged_df.iloc[0]['market_impact_score']}")
    message_lines.append(f"End-to-End Processing Latency: {merged_df.iloc[0]['latency']} seconds")

    message_lines.append("")
    message_lines.append(str(merged_df.iloc[0]["content"]))

    message_lines.append("")
    message_lines.append("ETF Signals:")

    for _, row in merged_df.iterrows():
        line = (
            f" - {row['symbol']}: {row['predicted_signal']}"
            f" | ML Model Accuracy={row['model_accuracy']:.1%}"
            f" | LLM Reasonableness Score={row['reasonableness_score']}"
            f" | Reason={row['brief_reason']}"
        )
        message_lines.append(line)

    message = "\n".join(message_lines)

    response = sns_client.publish(
        TopicArn=topic_arn,
        Subject=subject,
        Message=message,
    )

    return response



def publish_weekly_performance_to_sns(
    overall_df: pd.DataFrame,
    symbol_level_df: pd.DataFrame,
    buy_level_df: pd.DataFrame,
    sell_level_df: pd.DataFrame,
    reasonableness_level_df: pd.DataFrame,
    topic_arn: str | None = None,
    subject: str = "Weekly ETF Signal Performance Review",
    region_name: str = "us-east-1",
) -> dict:
    sns_client = boto3.client("sns", region_name=region_name)

    message_lines = []
    message_lines.append("Overview of last week's published ETF signal predictions.")
    message_lines.append("")

    overall_row = overall_df.iloc[0]
    message_lines.append("Overall Performance:")
    message_lines.append(
        f" - Total Signals={int(overall_row['total_signals'])}"
        f" | 30M Accuracy={overall_row['accuracy_30m']:.1%}"
        f" | Average 30M Return={overall_row['avg_30m_return']:.3f}%"
        f" | Potential Total 30M Return={overall_row['potential_total_30m_return']:.3f}%"
    )

    message_lines.append("")
    message_lines.append("Performance by Symbol:")

    for _, row in symbol_level_df.iterrows():
        message_lines.append(
            f" - {row['symbol']}"
            f" | Total Signals={int(row['total_signals'])}"
            f" | 30M Accuracy={row['accuracy_30m']:.1%}"
            f" | Average 30M Return={row['avg_30m_return']:.3f}%"
            f" | Potential Total 30M Return={row['potential_total_30m_return']:.3f}%"
        )

    buy_row = buy_level_df.iloc[0]
    message_lines.append("")
    message_lines.append("Buy Signals Performance:")
    message_lines.append(
        f" - Total Signals={int(buy_row['total_signals'])}"
        f" | 30M Accuracy={buy_row['accuracy_30m']:.1%}"
        f" | Average 30M Return={buy_row['avg_30m_return']:.3f}%"
        f" | Potential Total 30M Return={buy_row['potential_total_30m_return']:.3f}%"
    )

    sell_row = sell_level_df.iloc[0]
    message_lines.append("")
    message_lines.append("Sell Signals Performance:")
    message_lines.append(
        f" - Total Signals={int(sell_row['total_signals'])}"
        f" | 30M Accuracy={sell_row['accuracy_30m']:.1%}"
        f" | Average 30M Return={sell_row['avg_30m_return']:.3f}%"
        f" | Potential Total 30M Return={sell_row['potential_total_30m_return']:.3f}%"
    )

    reason_row = reasonableness_level_df.iloc[0]
    message_lines.append("")
    message_lines.append("High-Reasonableness Signals Performance:")
    message_lines.append(
        f" - Total Signals={int(reason_row['total_signals'])}"
        f" | 30M Accuracy={reason_row['accuracy_30m']:.1%}"
        f" | Average 30M Return={reason_row['avg_30m_return']:.3f}%"
        f" | Potential Total 30M Return={reason_row['potential_total_30m_return']:.3f}%"
    )

    message = "\n".join(message_lines)

    response = sns_client.publish(
        TopicArn=topic_arn,
        Subject=subject,
        Message=message,
    )

    return response