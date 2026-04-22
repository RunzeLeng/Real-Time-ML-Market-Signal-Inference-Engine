import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import json
import re
from etf_historical import post_etf_joins
from ml_modeling import train_xgboost_classifier, evaluate_xgboost_classifier
from prompt.standard_metrics import STANDARD_METRICS
from datetime import datetime


def extract_last_json_object(text: str) -> dict | None:
    candidates = []
    start_positions = [i for i, ch in enumerate(text) if ch == "{"]

    for start in start_positions:
        brace_count = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                brace_count += 1
            elif text[end] == "}":
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[start:end + 1]
                    try:
                        candidates.append(json.loads(candidate))
                    except Exception:
                        pass
                    break

    return candidates[-1] if candidates else None



def extract_last_json_object_with_keyword(text: str, json_keyword: str) -> dict | None:
    candidates = []
    start_positions = [i for i, ch in enumerate(text) if ch == "{"]

    for start in start_positions:
        brace_count = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                brace_count += 1
            elif text[end] == "}":
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[start:end + 1]
                    try:
                        candidates.append(json.loads(candidate))
                    except Exception:
                        pass
                    break
    
    candidates_with_keyword = [
        candidate for candidate in candidates
        if json_keyword in candidate
    ]

    return candidates_with_keyword[-1] if candidates_with_keyword else None



def load_batch_output_jsonl_to_df(folder_path: str = ".") -> pd.DataFrame:
    rows = []

    for file_path in sorted(Path(folder_path).glob("batch_finish*.jsonl")):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    model_output = str(record.get("model_output", "")).strip()

                    json_match = re.search(r"\{.*\}", model_output, re.DOTALL)

                    if json_match:
                        explanation_text = model_output[:json_match.start()].strip()
                        json_output = json_match.group(0).strip()
                    else:
                        explanation_text = model_output
                        json_output = None
                        
                    parsed_json = extract_last_json_object(json_output)

                    rows.append({
                        "id": record.get("id"),
                        "user_prompt": record.get("user_prompt"),
                        "explanation_text": explanation_text,
                        "json_output": json.dumps(parsed_json, ensure_ascii=False) if parsed_json is not None else None,
                    })

    return pd.DataFrame(rows, columns=["id", "user_prompt", "explanation_text", "json_output"])



def load_single_output_to_df(result: list[dict]) -> pd.DataFrame:
    if not result:
        return pd.DataFrame(columns=["id", "user_prompt", "explanation_text", "json_output"])

    rows = []
    record = result[0]
    model_output = str(record.get("model_output", "")).strip()

    json_match = re.search(r"\{.*\}", model_output, re.DOTALL)

    if json_match:
        explanation_text = model_output[:json_match.start()].strip()
        json_output = json_match.group(0).strip()
    else:
        explanation_text = model_output
        json_output = None
        
    parsed_json = extract_last_json_object(json_output)

    rows.append({
        "id": record.get("id"),
        "user_prompt": record.get("user_prompt"),
        "explanation_text": explanation_text,
        "json_output": json.dumps(parsed_json, ensure_ascii=False) if parsed_json is not None else None,
    })
    
    return pd.DataFrame(rows, columns=["id", "user_prompt", "explanation_text", "json_output"])



def convert_validator_output_to_df(output_text: str) -> pd.DataFrame:
    cleaned_text = output_text.strip().replace("```json", "").replace("```", "").strip()
    parsed_output = json.loads(cleaned_text)
    if parsed_output is None:
        raise ValueError("No valid JSON object found in validator output")

    market_impact_score = parsed_output.get("market_impact_score")
    signal_evaluations = parsed_output.get("signal_evaluations", [])

    rows = []
    for item in signal_evaluations:
        rows.append({
            "symbol": item.get("symbol"),
            "predicted_signal": item.get("predicted_signal"),
            "reasonableness_score": item.get("reasonableness_score"),
            "brief_reason": item.get("brief_reason"),
            "market_impact_score": market_impact_score,
        })

    return pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "predicted_signal",
            "reasonableness_score",
            "brief_reason",
            "market_impact_score",
        ],
    )



def convert_model_selection_output_to_df(output_text: str) -> pd.DataFrame:
    cleaned_text = output_text.strip().replace("```json", "").replace("```", "").strip()
    parsed_output = json.loads(cleaned_text)

    if parsed_output is None:
        raise ValueError("No valid JSON object found in model selection output")

    rows = []
    for _, item in parsed_output.items():
        rows.append({
            "symbol": item.get("symbol"),
            "combo_id": item.get("combo_id"),
            "reason": item.get("reason"),
        })

    return pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "combo_id",
            "reason",
        ],
    )



def convert_topic_summary_output_to_df(
    topic: str,
    processing_date: str,
    output_text: str,
) -> pd.DataFrame:
    parsed_output = extract_last_json_object_with_keyword(output_text, "overall_summary")

    if parsed_output is None:
        raise ValueError("No valid JSON object found in topic summary output")

    row = {
        "topic": topic,
        "processing_date": processing_date,
        "overall_summary": parsed_output.get("overall_summary", ""),
        "seven_day_summary": parsed_output.get("seven_day_summary", ""),
        "three_day_summary": parsed_output.get("three_day_summary", ""),
    }

    return pd.DataFrame(
        [row],
        columns=[
            "topic",
            "processing_date",
            "overall_summary",
            "seven_day_summary",
            "three_day_summary",
        ],
    )



def convert_news_topic_matching_output_to_df(
    output_text: str,
    uuid: str,
    title: str,
    published_at: datetime,
    source: str,
) -> pd.DataFrame:
    parsed_output = extract_last_json_object_with_keyword(output_text, "matched_topics")

    if parsed_output is None:
        raise ValueError("No valid JSON object found in topic matching output")

    matched_topics = parsed_output.get("matched_topics", [])

    rows = []

    for item in matched_topics:
        rows.append({
            "uuid": uuid,
            "title": title,
            "published_at": published_at,
            "source": source,
            "topic": item.get("topic_name"),
            "confidence_score": item.get("confidence_score"),
            "reason": item.get("reason"),
        })

    return pd.DataFrame(
        rows,
        columns=[
            "uuid",
            "title",
            "published_at",
            "source",
            "topic",
            "confidence_score",
            "reason",
        ],
    )



def convert_post_topic_matching_output_to_df(
    output_text: str,
    id: str,
    post: str,
) -> pd.DataFrame:
    parsed_output = extract_last_json_object_with_keyword(output_text, "matched_topics")

    if parsed_output is None:
        raise ValueError("No valid JSON object found in topic matching output")

    matched_topics = parsed_output.get("matched_topics", [])

    rows = []

    for item in matched_topics:
        rows.append({
            "id": id,
            "post": post,
            "topic": item.get("topic_name"),
            "confidence_score": item.get("confidence_score"),
            "reason": item.get("reason"),
        })

    return pd.DataFrame(
        rows,
        columns=[
            "id",
            "post",
            "topic",
            "confidence_score",
            "reason",
        ],
    )



def expand_json_output_to_metric_columns(
    df: pd.DataFrame,
    standard_metrics: dict,
) -> pd.DataFrame:
    expanded_rows = []

    for _, row in df.iterrows():
        metric_row = standard_metrics.copy()

        json_output = row.get("json_output")
        if json_output:
            metric_values = json.loads(json_output)
            
            filtered_metric_values = {
            key: value
            for key, value in metric_values.items()
            if key in standard_metrics
            }
            
            metric_row.update(filtered_metric_values)

        expanded_rows.append(metric_row)

    metrics_df = pd.DataFrame(expanded_rows)

    return pd.concat(
        [
            df[["id", "user_prompt", "explanation_text", "json_output"]].reset_index(drop=True),
            metrics_df.reset_index(drop=True),
        ],
        axis=1,
    )



def join_etf_and_json_output(
    etf_df: pd.DataFrame,
    json_df: pd.DataFrame,
) -> pd.DataFrame:
    json_columns_to_keep = [
        "id",
        "explanation_text",
        *[
            col for col in json_df.columns
            if col not in {"user_prompt", "json_output", "explanation_text"}
            and col not in etf_df.columns
        ],
    ]
    etf_df = etf_df.drop(columns=["timestamp", "created_at"]).copy()
    etf_df = etf_df.rename(columns={"created_at_seconds": "created_at"})

    json_subset = json_df[json_columns_to_keep].copy()

    joined_df = etf_df.merge(
        json_subset,
        how="inner",
        on="id",
    )

    return joined_df



def summarize_high_and_low_impact_metrics(
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
    df: pd.DataFrame,
    standard_metrics: dict,
) -> pd.DataFrame:
    scaled_df = df.copy()

    for col in standard_metrics.keys():
        if col in scaled_df.columns:
            scaled_df[f"x_{col}"] = pd.to_numeric(scaled_df[col], errors="coerce") / 4.0

    return scaled_df



def keep_only_x_and_y_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[[col for col in df.columns if col.startswith("x_") or col.startswith("y_")]].copy()



if __name__ == "__main__":
    load_dotenv()
    output_df = load_batch_output_jsonl_to_df()

    output_df = expand_json_output_to_metric_columns(output_df, STANDARD_METRICS)
    print(output_df)
    
    etf_df = post_etf_joins(
    bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
    post_object_key=os.getenv("AWS_S3_OBJECT_KEY_POST"),
    etf_object_key=os.getenv("AWS_S3_OBJECT_KEY_ETF"),
    post_start_date="2025-01-01",
    post_end_date="2025-12-31",
    min_content_length=50,
    post_duplicate=False,
    )
    # df = df.head(200)
    # etf_df = etf_df.iloc[:1075]
    print(etf_df)
    
    df = join_etf_and_json_output(etf_df, output_df)
    
    df = add_categorical_target_columns(df, target_config=("QQQ", -0.09, 0.09, "30m", 2))
    df = scale_input_metric_columns(df, STANDARD_METRICS)
    df = keep_only_x_and_y_columns(df)
    
    # top_20_df, bottom_20_df = summarize_high_and_low_impact_metrics(df, STANDARD_METRICS, top_n=20)
    print(df)
    
    
    train_accuracies = []
    valid_accuracies = []
    accuracy_diffs = []
    lower_thresholds = []
    upper_thresholds = []
    valid_accuracies_high_confidence = []
    high_confidence_percentages = []
    
    for i in range(300, 500):
        model, X_train, X_valid, y_train, y_valid = train_xgboost_classifier(
        df=df,
        
        test_size=0.2,
        callback_early_stopping_rounds=50,
        
        max_depth=5,
        
        subsample=0.6,
        colsample_bytree=0.6,
        colsample_bylevel=0.8,
        colsample_bynode=1.0,
        
        min_child_weight=1,
        gamma=1,
        
        reg_alpha=0.0,
        reg_lambda=2.0,
        learning_rate=0.05,
        
        scale_pos_weight=1.2,
        
        objective="binary:logistic",
        eval_metric="logloss",
        num_classes= 2,
        random_state=i,
        )
        
        (
            train_accuracy_customized, 
            valid_accuracy_customized, 
            accuracy_diff,
            lower_threshold, 
            upper_threshold, 
            valid_accuracy_high_confidence, 
            high_confidence_percentage 
        ) = evaluate_xgboost_classifier(
            model, 
            X_train,
            X_valid, 
            y_train, 
            y_valid, 
            num_classes=2
        )
        
        train_accuracies.append(train_accuracy_customized)
        valid_accuracies.append(valid_accuracy_customized)
        accuracy_diffs.append(accuracy_diff)
        
        lower_thresholds.append(lower_threshold)
        upper_thresholds.append(upper_threshold)
        valid_accuracies_high_confidence.append(valid_accuracy_high_confidence)
        high_confidence_percentages.append(high_confidence_percentage)
        
        lower_thresholds = [x for x in lower_thresholds if x is not None]
        upper_thresholds = [x for x in upper_thresholds if x is not None]
        valid_accuracies_high_confidence = [x for x in valid_accuracies_high_confidence if x is not None]
        high_confidence_percentages = [x for x in high_confidence_percentages if x is not None]

        # model.save_model("output_xgboost_model.json")
    
    print("\nAverage Training Accuracy:", sum(train_accuracies) / len(train_accuracies))
    print("\nAverage Validation Accuracy:", sum(valid_accuracies) / len(valid_accuracies))
    print("\nAverage Accuracy Gap:", sum(accuracy_diffs) / len(accuracy_diffs))
    print("\nAverage Lower Threshold:", sum(lower_thresholds) / len(lower_thresholds))
    print("\nAverage Upper Threshold:", sum(upper_thresholds) / len(upper_thresholds))
    print("\nAverage Validation Accuracy on High-Confidence Subset:", sum(valid_accuracies_high_confidence) / len(valid_accuracies_high_confidence))
    print("\nAverage High-Confidence Subset Percentage:", sum(high_confidence_percentages) / len(high_confidence_percentages))
