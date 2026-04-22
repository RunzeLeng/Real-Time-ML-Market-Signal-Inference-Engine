from pathlib import Path
import pandas as pd
from aws_aurora_dsql import create_table_and_load_df_to_aurora, dsql_execute_sql
from aws_bedrock import query_bedrock_model
from ml_training_data_building import convert_post_topic_matching_output_to_df, convert_topic_summary_output_to_df
from dotenv import load_dotenv
import os


def match_post_to_topics(id: str, post: str) -> pd.DataFrame:
    
    system_prompt = Path("src/prompt/text_to_topic_system_prompt.txt").read_text(encoding="utf-8")
    user_prompt = f"Title: {str(post).strip()}"

    output_text = query_bedrock_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-1",
        temperature=0.2,
        max_tokens=600,
        top_p=0.95,
        top_k=250,
        system_prompt_caching=True,
    )

    post_topic_match_df = convert_post_topic_matching_output_to_df(
        output_text=output_text,
        id=id,
        post=post,
    )

    return post_topic_match_df



def summarize_news_by_topic(
    news_df: pd.DataFrame,
    processing_date: str,
) -> pd.DataFrame:

    system_prompt = Path("src/prompt/topic_summary_system_prompt.txt").read_text(encoding="utf-8")
    user_prompt = f"Processing date: {processing_date}"

    all_result_dfs = []

    topics = (
        news_df["topic"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("")]
        .drop_duplicates()
        .tolist()
    )
    
    csv_columns = [
        "title",
        "published_at",
        "source",
        "topic",
        "confidence_score",
        "reason",
    ]

    for topic in topics:
        filtered_topic_df = (
            news_df[news_df["topic"].astype(str).str.strip() == topic]
            .copy()
            .reset_index(drop=True)
        )
        
        filtered_topic_df = filtered_topic_df[csv_columns].copy()
        csv_bytes = filtered_topic_df.to_csv(index=False).encode("utf-8")

        model_output = query_bedrock_model(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name="us-east-1",
            temperature=0.2,
            max_tokens=1000,
            top_p=0.95,
            top_k=250,
            system_prompt_caching=True,
            include_document=True,
            document_format="csv",
            document_name=f"{topic}_news",
            document_bytes=csv_bytes,
        )

        topic_summary_df = convert_topic_summary_output_to_df(
            topic=topic,
            processing_date=processing_date,
            output_text=model_output,
        )

        all_result_dfs.append(topic_summary_df)

    return pd.concat(all_result_dfs, ignore_index=True)



if __name__ == "__main__":
    load_dotenv()
        
    # result_df = match_post_to_topics(
    #     id="123",
    #     post="Trump threatens new tariffs on China after failed negotiations.",
    # )
    
    # print(result_df)
    
    sql_query_2 = """
        SELECT * FROM training_data.news_topic_matching
    """
    
    rows = dsql_execute_sql(
        host=os.getenv("AWS_AURORA_DB_HOST"),
        database="postgres",
        sql=sql_query_2,
        user="admin",
        region="us-east-1",
        profile="default",
    )
    print(rows)
    
    summary_df = summarize_news_by_topic(news_df=rows, processing_date="2026-04-15")
    print(summary_df)
    
    create_table_and_load_df_to_aurora(
        df=summary_df,
        host=os.getenv("AWS_AURORA_DB_HOST"),
        database="postgres",
        schema_name="training_data",
        table_name="topic_summary",
        create_table=False,
    )