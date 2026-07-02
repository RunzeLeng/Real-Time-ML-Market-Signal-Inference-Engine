import boto3
import pandas as pd
from src.common.config import SnsConfig, config


class SnsNotificationService:
    
    def __init__(
        self,
        sns_config: SnsConfig | None = None,
    ) -> None:
        self.config = sns_config or config.sns
        self.client = boto3.client("sns", region_name=self.config.region)
    
    
    
    def publish_etf_signals(
        self,
        merged_df: pd.DataFrame,
        subject: str = "ETF Market Signal Alert",
    ) -> dict:
        
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

        return self._publish(
            message=message,
            subject=subject,
        )
    
    
    
    def publish_weekly_performance(
        self,
        overall_df: pd.DataFrame,
        symbol_level_df: pd.DataFrame,
        buy_level_df: pd.DataFrame,
        sell_level_df: pd.DataFrame,
        reasonableness_level_df: pd.DataFrame,
        subject: str = "Weekly ETF Signal Performance Review",
    ) -> dict:
        
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

        return self._publish(
            message=message,
            subject=subject,
        )
    
    
    
    def _publish(
        self,
        message: str,
        subject: str,
    ) -> dict:
    
        return self.client.publish(
            TopicArn=self.config.topic_arn,
            Subject=subject,
            Message=message,
        )