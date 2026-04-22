from etf_constants import ETF_LIST
from aws_aurora_dsql import create_table_and_load_df_to_aurora, dsql_execute_sql
import os
from dotenv import load_dotenv
import pandas as pd
from aws_bedrock import query_bedrock_model
from pathlib import Path
from ml_model_auto_optimizer import ml_model_automatic_optimizer
from ml_model_deployment import save_selected_models
from ml_training_data_building import convert_model_selection_output_to_df


def load_model_performance_by_training_version(training_version: str) -> pd.DataFrame:
    sql_query = f"""
        select * from training_output.model_performance_{training_version}
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



def llm_auto_model_selection(training_version: str, ECS_ETF_LIST_OVERRIDE: list | None = None) -> None:
    load_dotenv()
    
    model_performance = load_model_performance_by_training_version(
        training_version=training_version
    )
    
    if ECS_ETF_LIST_OVERRIDE:
        etf_list = ECS_ETF_LIST_OVERRIDE
    else:
        etf_list = ETF_LIST
    
    for symbol in etf_list:
        
        filtered_model_performance = model_performance[model_performance["symbol"] == symbol].copy()
        print(f"Selecting symbol: {symbol}")
        
        csv_bytes = filtered_model_performance.to_csv(index=False).encode("utf-8")
        system_prompt = Path("src/prompt/model_selection_system_prompt.txt").read_text(encoding="utf-8")
        user_prompt = Path("src/prompt/model_selection_user_prompt.txt").read_text(encoding="utf-8")
        
        model_selection_result = query_bedrock_model(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_id="us.anthropic.claude-opus-4-6-v1",
            region_name="us-east-1",
            temperature=0.2,
            max_tokens=1000,
            top_p=0.95,
            top_k=100,
            system_prompt_caching=False,
            include_document=True,
            document_format="csv",
            document_name="attached_file",
            document_bytes=csv_bytes,
        )
        
        model_selection_df = convert_model_selection_output_to_df(model_selection_result)
        
        create_table_and_load_df_to_aurora(
            df=model_selection_df,
            host=os.getenv("AWS_AURORA_DB_HOST"),
            database="postgres",
            schema_name="training_output",
            table_name=f"selected_models_{training_version}",
            create_table=(symbol == "QQQ"),
        )



def ml_model_auto_retraining_pipeline(training_version: str, ECS_ETF_LIST_OVERRIDE: list | None = None) -> None:
    
    ml_model_automatic_optimizer(
        training_version=training_version, 
        ECS_ETF_LIST_OVERRIDE=ECS_ETF_LIST_OVERRIDE
    )
    
    llm_auto_model_selection(
        training_version=training_version, 
        ECS_ETF_LIST_OVERRIDE=ECS_ETF_LIST_OVERRIDE
    )
    
    save_selected_models(
        training_version=training_version, 
        ECS_ETF_LIST_OVERRIDE=ECS_ETF_LIST_OVERRIDE
    )



def get_ecs_etf_list_override() -> list[str] | None:
    target_etfs_env = os.getenv("TARGET_ETFS", "").strip()

    if not target_etfs_env:
        return None

    ecs_etf_list_override = [
        etf.strip().upper()
        for etf in target_etfs_env.split(",")
        if etf.strip()
    ]

    return ecs_etf_list_override



if __name__ == "__main__":
    
    load_dotenv()
    
    ECS_ETF_LIST_OVERRIDE = get_ecs_etf_list_override()
    
    ml_model_auto_retraining_pipeline(
        training_version="v2",
        ECS_ETF_LIST_OVERRIDE=ECS_ETF_LIST_OVERRIDE,
    )