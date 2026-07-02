import pandas as pd
from src.market_data.etf_market_data import EtfMarketDataService
from src.processing.json_processing import JSONProcessingService
from src.prompt.standard_metrics import STANDARD_METRICS



class TrainingDataBuilder:


    def summarize_high_and_low_impact_metrics(
        self,
        df: pd.DataFrame,
        standard_metrics: dict,
        top_n: int = 20,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        metric_cols = [col for col in standard_metrics.keys() if col in df.columns]

        numeric_df = df[metric_cols].apply(pd.to_numeric, errors="coerce")
        column_sums = numeric_df.sum().sort_values(ascending=False)

        top_df = column_sums.head(top_n).reset_index()
        top_df.columns = ["metric", "sum"]
        print("Top Metrics:\n", top_df)

        bottom_df = column_sums.tail(top_n).reset_index()
        bottom_df.columns = ["metric", "sum"]
        print("\nBottom Metrics:\n", bottom_df)

        return top_df, bottom_df



    def add_categorical_target_columns(
        self,
        df: pd.DataFrame,
        target_config: tuple[str, float, float, str, int],
    ) -> pd.DataFrame:
        
        categorized_df = df.copy()
        symbol, lower_threshold, upper_threshold, timeframe, num_classes = target_config

        if timeframe.lower() == "all":
            timeframes_to_process = ["5m", "10m", "30m", "1h", "3h"]
        else:
            timeframes_to_process = [timeframe.lower()]

        def categorize_value_4_class(x):
            if pd.isna(x):
                return None
            if x > upper_threshold:
                return "strong_buy"
            elif 0 <= x <= upper_threshold:
                return "buy"
            elif lower_threshold <= x < 0:
                return "sell"
            else:
                return "strong_sell"

        def categorize_value_3_class(x):
            if pd.isna(x):
                return None
            if x > upper_threshold:
                return "buy"
            elif lower_threshold <= x <= upper_threshold:
                return "hold"
            else:
                return "sell"

        for tf in timeframes_to_process:
            matching_col = None

            for col in categorized_df.columns:
                if tf in col.lower() and not col.startswith("y_"):
                    matching_col = col
                    break

            if matching_col is None:
                raise ValueError(f"No source column found for timeframe: {tf}")

            target_column_name = f"y_{symbol.lower()}_{tf}"

            if num_classes == 4:
                categorized_df[target_column_name] = categorized_df[matching_col].apply(categorize_value_4_class)
                
            elif num_classes == 3:
                categorized_df[target_column_name] = categorized_df[matching_col].apply(categorize_value_3_class)
                
            else:
                categorized_df[target_column_name] = categorized_df[matching_col].apply(categorize_value_3_class)
                categorized_df = categorized_df[categorized_df[target_column_name] != "hold"].copy()

            categorized_df = categorized_df.dropna(subset=[target_column_name])

        return categorized_df



    def scale_input_metric_columns(
        self,
        df: pd.DataFrame,
        standard_metrics: dict,
    ) -> pd.DataFrame:
        scaled_df = df.copy()

        for col in standard_metrics.keys():
            if col in scaled_df.columns:
                scaled_df[f"x_{col}"] = pd.to_numeric(scaled_df[col], errors="coerce") / 4.0

        return scaled_df



    def keep_only_x_and_y_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[[col for col in df.columns if col.startswith("x_") or col.startswith("y_")]].copy()



    def load_training_data(
        self,
        post_start_date: str = "2025-01-01",
        post_end_date: str = "2026-03-25",
    ) -> pd.DataFrame:
        
        json_processing_service = JSONProcessingService()
        
        output_df = json_processing_service.load_batch_output_to_df()
        
        output_df = json_processing_service.expand_metric_json_to_columns(output_df, STANDARD_METRICS)
        
        etf_market_data_service = EtfMarketDataService()
        etf_df = etf_market_data_service.join_posts_with_etf_features(
            post_start_date=post_start_date,
            post_end_date=post_end_date,
            min_content_length=50,
            post_duplicate=False,
        )
        
        df = json_processing_service.join_etf_with_json_output(etf_df, output_df)
        
        print(f"\nJSON output row count matches unique post ID count: {len(output_df) == etf_df['id'].nunique()}")
        print(f"\nPost_etf_joins row count matches JSON_output_etf_joins row count: {len(etf_df) == len(df)}")
        print(f"\nDF row count for processing is: {len(df)}")

        return df
