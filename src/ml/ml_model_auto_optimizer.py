import os
from dotenv import load_dotenv
import random
import pandas as pd
from etf_historical import post_etf_joins
from ml_modeling import train_xgboost_classifier, evaluate_xgboost_classifier
from prompt.standard_metrics import STANDARD_METRICS
from ml_training_data_building import *
import itertools
from aws_aurora_dsql import create_table_and_load_df_to_aurora
from etf_constants import ETF_LIST


def get_scale_pos_weight_ratio(
    df: pd.DataFrame,
    prediction_range: str,
) -> float | None:
    matching_cols = [col for col in df.columns if prediction_range.lower() in col.lower()]
    if not matching_cols:
        raise ValueError(f"No column found containing prediction range: {prediction_range}")

    target_col = matching_cols[0]
    values = pd.to_numeric(df[target_col], errors="coerce").dropna()

    negative_count = (values < 0).sum()
    positive_count = (values > 0).sum()

    if positive_count == 0:
        return None

    return negative_count / positive_count



def model_training_optimizer(
    df: pd.DataFrame,
    symbol: str,
    STANDARD_METRICS: dict,
    random_state_length: int = 100,
    prediction_range: str = "30m",
) -> pd.DataFrame:
    complexity_group = [
        {"max_depth": 4, "min_child_weight": 3, "gamma": 2},
        {"max_depth": 5, "min_child_weight": 1, "gamma": 1},
        {"max_depth": 7, "min_child_weight": 1, "gamma": 0},
    ]

    sampling_group = [
        {"subsample": 0.5, "colsample_bytree": 0.5, "colsample_bylevel": 0.7, "colsample_bynode": 0.8},
        {"subsample": 0.6, "colsample_bytree": 0.6, "colsample_bylevel": 0.8, "colsample_bynode": 1.0},
        {"subsample": 0.8, "colsample_bytree": 0.8, "colsample_bylevel": 1.0, "colsample_bynode": 1.0},
    ]

    shrinkage_group = [
        {"learning_rate": 0.03, "reg_lambda": 3.0, "reg_alpha": 0.2},
        {"learning_rate": 0.05, "reg_lambda": 2.0, "reg_alpha": 0.0},
        {"learning_rate": 0.08, "reg_lambda": 1.0, "reg_alpha": 0.0},
    ]
    
    base_scale_pos_weight = get_scale_pos_weight_ratio(df, prediction_range)

    class_imbalance_group = [
        {"scale_pos_weight": base_scale_pos_weight * 0.8},
        {"scale_pos_weight": base_scale_pos_weight},
        {"scale_pos_weight": base_scale_pos_weight * 1.2},
    ]

    hold_threshold_group = [0.05, 0.07, 0.09]

    results = []

    all_combinations = itertools.product(
        complexity_group,
        sampling_group,
        shrinkage_group,
        class_imbalance_group,
        hold_threshold_group,
    )

    for combo_id, (complexity, sampling, shrinkage, class_imbalance, holding_gap) in enumerate(all_combinations, start=1):
        print(f"Running combination {combo_id} with parameters and scale_pos_weight={class_imbalance['scale_pos_weight']}:")
        
        df_model = add_categorical_target_columns(df.copy(), target_config=(symbol, -holding_gap, holding_gap, prediction_range, 2))
        df_model = scale_input_metric_columns(df_model, STANDARD_METRICS)
        df_model = keep_only_x_and_y_columns(df_model)

        train_accuracies = []
        valid_accuracies = []
        accuracy_diffs = []
        
        lower_thresholds = []
        upper_thresholds = []
        average_thresholds = []
        
        valid_accuracies_high_confidence = []
        high_confidence_percentages = []
        
        random_number = random.randint(1, 1000)
        for i in range(random_number, random_number+random_state_length):
            model, X_train, X_valid, y_train, y_valid = train_xgboost_classifier(
                df=df_model,
                test_size=0.2,
                callback_early_stopping_rounds=50,
                
                max_depth=complexity["max_depth"],
                min_child_weight=complexity["min_child_weight"],
                gamma=complexity["gamma"],

                subsample=sampling["subsample"],
                colsample_bytree=sampling["colsample_bytree"],
                colsample_bylevel=sampling["colsample_bylevel"],
                colsample_bynode=sampling["colsample_bynode"],

                learning_rate=shrinkage["learning_rate"],
                reg_lambda=shrinkage["reg_lambda"],
                reg_alpha=shrinkage["reg_alpha"],
            
                scale_pos_weight=class_imbalance["scale_pos_weight"],

                objective="binary:logistic",
                eval_metric="logloss",
                num_classes=2,
                random_state=i,
            )

            (
                train_accuracy_customized,
                valid_accuracy_customized,
                accuracy_diff,
                lower_threshold,
                upper_threshold,
                valid_accuracy_high_confidence,
                high_confidence_percentage,
                save_model_to_s3,
            ) = evaluate_xgboost_classifier(
                model,
                X_train,
                X_valid,
                y_train,
                y_valid,
                num_classes=2,
                if_print_results=False,
            )

            train_accuracies.append(train_accuracy_customized)
            valid_accuracies.append(valid_accuracy_customized)
            accuracy_diffs.append(accuracy_diff)

            if lower_threshold is not None:
                lower_thresholds.append(lower_threshold)
            if upper_threshold is not None:
                upper_thresholds.append(upper_threshold)
            if lower_threshold is not None and upper_threshold is not None:
                average_thresholds.append((lower_threshold + upper_threshold) / 2.0)
            if valid_accuracy_high_confidence is not None:
                valid_accuracies_high_confidence.append(valid_accuracy_high_confidence)
            if high_confidence_percentage is not None:
                high_confidence_percentages.append(high_confidence_percentage)

        result_row = {
            "symbol": symbol,
            
            "combo_id": combo_id,

            "holding_gap": holding_gap,

            "max_depth": complexity["max_depth"],
            "min_child_weight": complexity["min_child_weight"],
            "gamma": complexity["gamma"],

            "subsample": sampling["subsample"],
            "colsample_bytree": sampling["colsample_bytree"],
            "colsample_bylevel": sampling["colsample_bylevel"],
            "colsample_bynode": sampling["colsample_bynode"],

            "learning_rate": shrinkage["learning_rate"],
            "reg_lambda": shrinkage["reg_lambda"],
            "reg_alpha": shrinkage["reg_alpha"],

            "scale_pos_weight": class_imbalance["scale_pos_weight"],

            "avg_train_accuracy": sum(train_accuracies) / len(train_accuracies),
            "avg_valid_accuracy": sum(valid_accuracies) / len(valid_accuracies),
            "avg_accuracy_gap": sum(accuracy_diffs) / len(accuracy_diffs),

            "avg_lower_threshold": sum(lower_thresholds) / len(lower_thresholds) if lower_thresholds else None,
            "avg_upper_threshold": sum(upper_thresholds) / len(upper_thresholds) if upper_thresholds else None,
            "avg_threshold": sum(average_thresholds) / len(average_thresholds) if average_thresholds else None,

            "avg_valid_accuracy_high_confidence": (
                sum(valid_accuracies_high_confidence) / len(valid_accuracies_high_confidence)
                if valid_accuracies_high_confidence else None
            ),
            "std_valid_accuracy_high_confidence": (
                pd.Series(valid_accuracies_high_confidence).std()
                if len(valid_accuracies_high_confidence) > 1 else None
            ),  
            "avg_high_confidence_coverage_percentage": (
                sum(high_confidence_percentages) / len(high_confidence_percentages)
                if high_confidence_percentages else None
            ),
            "pct_valid_accuracy_high_confidence": len(valid_accuracies_high_confidence)/random_state_length,
        }

        results.append(result_row)

    return pd.DataFrame(results)



def load_training_data(
    post_start_date: str = "2025-01-01",
    post_end_date: str = "2026-03-25",
) -> pd.DataFrame:
    
    output_df = load_batch_output_jsonl_to_df()
    
    output_df = expand_json_output_to_metric_columns(output_df, STANDARD_METRICS)
    
    etf_df = post_etf_joins(
    bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
    post_object_key=os.getenv("AWS_S3_OBJECT_KEY_POST"),
    etf_object_key=os.getenv("AWS_S3_OBJECT_KEY_ETF"),
    post_start_date=post_start_date,
    post_end_date=post_end_date,
    min_content_length=50,
    post_duplicate=False,
    )
    
    df = join_etf_and_json_output(etf_df, output_df)
    
    print(f"\nJSON output row count matches unique post ID count: {len(output_df) == etf_df['id'].nunique()}")
    print(f"\nPost_etf_joins row count matches JSON_output_etf_joins row count: {len(etf_df) == len(df)}")
    print(f"\nDF row count for processing is: {len(df)}")

    return df



def ml_model_automatic_optimizer(training_version: str, ECS_ETF_LIST_OVERRIDE: list | None = None) -> None:
    load_dotenv()
    
    df = load_training_data()
    
    if ECS_ETF_LIST_OVERRIDE:
        etf_list = ECS_ETF_LIST_OVERRIDE
    else:
        etf_list = ETF_LIST
    
    for symbol in etf_list:
        
        filtered_df = df[df["symbol"] == symbol].copy()
        print(f"\nSymbol: {symbol}, count: {len(filtered_df)}")
        
        training_results = model_training_optimizer(filtered_df, symbol, STANDARD_METRICS, random_state_length=100)
        
        create_table_and_load_df_to_aurora(
            df=training_results,
            host=os.getenv("AWS_AURORA_DB_HOST"),
            database="postgres",
            schema_name="training_output",
            table_name=f"model_performance_{training_version}",
            create_table=(symbol == "QQQ"),
        )
        
        

if __name__ == "__main__":
    
    ml_model_automatic_optimizer(training_version="v2")