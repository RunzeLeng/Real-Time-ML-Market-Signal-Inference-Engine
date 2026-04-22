import os
from dotenv import load_dotenv
import random
import pandas as pd
from ml_model_auto_optimizer import load_training_data
from ml_modeling import train_xgboost_classifier, evaluate_xgboost_classifier
from prompt.standard_metrics import STANDARD_METRICS
from ml_training_data_building import *
from aws_aurora_dsql import create_table_and_load_df_to_aurora, dsql_execute_sql
import tempfile
import boto3
from xgboost import XGBClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed
from etf_constants import ETF_LIST


def load_selected_model_performance() -> pd.DataFrame:
    sql_query = """
        select symbol, combo_id, avg_upper_threshold, avg_lower_threshold, prediction_range
        from training_output.selected_model_performance
    """
    
    rows = dsql_execute_sql(
        host=os.getenv("AWS_AURORA_DB_HOST"),
        database="postgres",
        sql=sql_query,
        user="admin",
        region="us-east-1",
        profile="default",
    )
    
    return rows



def load_xgboost_models_from_s3(prefix: str = "models", max_workers: int = 8) -> dict:
    bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("AWS_S3_BUCKET_NAME is missing")

    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    s3_keys = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            if s3_key.endswith(".json"):
                s3_keys.append(s3_key)

    def load_one_model(s3_key: str):
        file_name = os.path.basename(s3_key)
        model_name = file_name.removesuffix(".json")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_file_path = temp_file.name

        try:
            local_s3_client = boto3.client("s3")
            local_s3_client.download_file(bucket_name, s3_key, temp_file_path)

            model = XGBClassifier()
            model.load_model(temp_file_path)

            return model_name, model
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    loaded_models = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_one_model, s3_key) for s3_key in s3_keys]

        for future in as_completed(futures):
            model_name, model = future.result()
            loaded_models[model_name] = model

    return loaded_models



def save_xgboost_model_to_s3(
    model,
    symbol: str,
    combo_id: int,
    random_state: int,
    if_save_model: bool = False,
    prediction_range: str = "30m",
    prefix: str = "models",
) -> None:

    if if_save_model:
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("AWS_S3_BUCKET_NAME is missing")

        file_name = f"{symbol}_{prediction_range}_{combo_id}_{random_state}_XGBoost_Model.json"
        s3_key = f"{prefix}/{symbol}/{prediction_range}/{combo_id}/{file_name}"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_file_path = temp_file.name

        try:
            model.save_model(temp_file_path)

            s3_client = boto3.client("s3")
            s3_client.upload_file(temp_file_path, bucket_name, s3_key)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
                

def model_training_with_selected_hyperparameter_combinations(
    df: pd.DataFrame,
    model_combos: pd.DataFrame,
    symbol: str,
    STANDARD_METRICS: dict,
    random_state_length: int = 100,
    prediction_range: str = "30m",
    prefix: str = "models",
) -> pd.DataFrame:
    results = []
    
    for _, combo_row in model_combos.iterrows():
        print(f"\nRunning {combo_row['symbol']} combination {combo_row['combo_id']} with parameters and scale_pos_weight={combo_row['scale_pos_weight']}:")
        
        holding_gap = combo_row["holding_gap"]

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
                
                max_depth=combo_row["max_depth"],
                min_child_weight=combo_row["min_child_weight"],
                gamma=combo_row["gamma"],

                subsample=combo_row["subsample"],
                colsample_bytree=combo_row["colsample_bytree"],
                colsample_bylevel=combo_row["colsample_bylevel"],
                colsample_bynode=combo_row["colsample_bynode"],

                learning_rate=combo_row["learning_rate"],
                reg_lambda=combo_row["reg_lambda"],
                reg_alpha=combo_row["reg_alpha"],
            
                scale_pos_weight=combo_row["scale_pos_weight"],

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
            
            save_xgboost_model_to_s3(
                model,
                symbol=symbol,
                combo_id=combo_row["combo_id"],
                random_state=i,
                if_save_model=save_model_to_s3,
                prediction_range=prediction_range,
                prefix=prefix,
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
            
            "prediction_range": prediction_range,
            
            "combo_id": combo_row["combo_id"],

            "holding_gap": holding_gap,

            "max_depth": combo_row["max_depth"],
            "min_child_weight": combo_row["min_child_weight"],
            "gamma": combo_row["gamma"],

            "subsample": combo_row["subsample"],
            "colsample_bytree": combo_row["colsample_bytree"],
            "colsample_bylevel": combo_row["colsample_bylevel"],
            "colsample_bynode": combo_row["colsample_bynode"],

            "learning_rate": combo_row["learning_rate"],
            "reg_lambda": combo_row["reg_lambda"],
            "reg_alpha": combo_row["reg_alpha"],

            "scale_pos_weight": combo_row["scale_pos_weight"],

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



def load_selected_model_combos_from_aurora(training_version: str = "models") -> pd.DataFrame:
    if training_version == "models":
        sql_query = """
            SELECT B.* 
            FROM training_output.selected_models AS A
            INNER JOIN training_output.model_performance AS B
                ON A.combo_id = B.combo_id
                AND A.symbol = B.symbol
            ORDER BY A.symbol, A.combo_id;
        """
    else:
        sql_query = f"""
            SELECT B.* 
            FROM training_output.selected_models_{training_version} AS A
            INNER JOIN training_output.model_performance_{training_version} AS B
                ON A.combo_id = B.combo_id
                AND A.symbol = B.symbol
            ORDER BY A.symbol, A.combo_id;
        """
    
    model_combos = dsql_execute_sql(
        host=os.getenv("AWS_AURORA_DB_HOST"),
        database="postgres",
        sql=sql_query,
        user="admin",
        region="us-east-1",
        profile="default",
    )
    
    return model_combos



def save_selected_models(training_version: str = "models", ECS_ETF_LIST_OVERRIDE: list | None = None) -> None:
    load_dotenv()
    
    df = load_training_data()
    
    model_combos = load_selected_model_combos_from_aurora(
        training_version=training_version
    )
    
    if ECS_ETF_LIST_OVERRIDE:
        etf_list = ECS_ETF_LIST_OVERRIDE
    else:
        etf_list = ETF_LIST
    
    for symbol in etf_list:
        
        filtered_df = df[df["symbol"] == symbol].copy()
        filtered_model_combos = model_combos[model_combos["symbol"] == symbol].copy()
        
        print(f"\nSymbol: {symbol}, count: {len(filtered_df)}")
        print(f"\nSymbol: {symbol}, count: {len(filtered_model_combos)}")
        
        training_results = model_training_with_selected_hyperparameter_combinations(filtered_df, filtered_model_combos, symbol, STANDARD_METRICS, random_state_length=100, prefix=training_version)
        
        create_table_and_load_df_to_aurora(
            df=training_results,
            host=os.getenv("AWS_AURORA_DB_HOST"),
            database="postgres",
            schema_name="training_output",
            table_name=f"selected_model_performance_{training_version}",
            create_table=(symbol == "QQQ"),
        )
        
        

def model_predict_with_group_average(
    X,
    models: dict,
    symbol: str,
    combo_id: int,
    lower_threshold: float,
    upper_threshold: float,
    prediction_range: str = "30m",
) -> str | None:
    matching_models = [
        model
        for model_name, model in models.items()
        if model_name.startswith(f"{symbol}_{prediction_range}_{combo_id}_")
    ]

    if not matching_models:
        return None

    sell_probs = [
        model.predict_proba(X)[0, 0]
        for model in matching_models
    ]

    avg_prob = sum(sell_probs) / len(sell_probs)

    if avg_prob >= upper_threshold:
        return "sell"
    if avg_prob <= lower_threshold:
        return "buy"

    return None



def predict_symbol_combo_signals(
    X,
    models: dict,
    selected_model_df: pd.DataFrame,
    max_workers: int = 8,
) -> pd.DataFrame:
    def run_one_combo(row):
        symbol = row["symbol"]
        combo_id = row["combo_id"]
        lower_threshold = row["avg_lower_threshold"]
        upper_threshold = row["avg_upper_threshold"]
        prediction_range = row["prediction_range"]

        signal = model_predict_with_group_average(
            X=X,
            models=models,
            symbol=symbol,
            combo_id=combo_id,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            prediction_range=prediction_range,
        )

        return {
            "symbol": symbol,
            "combo_id": combo_id,
            "signal": signal,
        }

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_one_combo, row)
            for _, row in selected_model_df.iterrows()
        ]

        for future in as_completed(futures):
            results.append(future.result())

    return pd.DataFrame(results)



def symbol_voting_system(signal_df: pd.DataFrame, id_df: pd.DataFrame) -> pd.DataFrame:
    vote_map = {
        "sell": 1,
        "buy": -1,
        None: 0,
    }

    working_df = signal_df.copy()
    working_df["vote"] = working_df["signal"].map(vote_map).fillna(0)

    symbol_votes = (
        working_df.groupby("symbol", as_index=False)["vote"]
        .sum()
        .rename(columns={"vote": "vote_sum"})
    )

    symbol_votes["final_signal"] = symbol_votes["vote_sum"].apply(
        lambda x: "sell" if x >= 3 else "buy" if x <= -3 else None
    )

    output_df = symbol_votes[["symbol", "final_signal"]].copy()
    output_df["id"] = id_df.iloc[0]["id"]

    return output_df[["id", "symbol", "final_signal"]]



def merge_post_signal_and_validation_dfs(
    signal_prediction: pd.DataFrame,
    post_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> pd.DataFrame:
    filtered_signal_prediction = signal_prediction[
        signal_prediction["final_signal"].notna()
    ].copy()

    prepared_post_df = post_df.drop(columns=["created_at"], errors="ignore").rename(
        columns={"created_at_seconds": "created_at"}
    )

    merged_df = prepared_post_df.merge(
        filtered_signal_prediction,
        on="id",
        how="inner",
    )

    merged_df = merged_df.merge(
        validation_df,
        left_on=["symbol", "final_signal"],
        right_on=["symbol", "predicted_signal"],
        how="inner",
    )
    
    merged_df["combined_score"] = (
        merged_df["reasonableness_score"] * merged_df["market_impact_score"]
    )
    
    return merged_df[
        [
        "id",
        "created_at",
        "content",
        "symbol",
        "predicted_signal",
        "market_impact_score",
        "reasonableness_score",
        "brief_reason",
        "combined_score",
        ]
    ]



def score_decision_layer(
    df: pd.DataFrame,
    symbol_threshold: float = 0.7,
    combined_threshold: float = 0.4,
) -> pd.DataFrame:
    scored_df = df.copy()
    
    scored_df = scored_df[
        (scored_df["reasonableness_score"] >= symbol_threshold)
        & (scored_df["combined_score"] >= combined_threshold)
    ].copy()

    return scored_df



def calculate_processing_latency(df: pd.DataFrame) -> pd.DataFrame:
    latency_df = df.copy()

    current_time = pd.Timestamp.now(tz="US/Eastern")
    created_at_time = pd.to_datetime(latency_df.iloc[0]["created_at"])

    latency_df["latency"] = round((current_time - created_at_time).total_seconds(), 2)
    
    return latency_df



if __name__ == "__main__":
    
    save_selected_models()