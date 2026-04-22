from pathlib import Path
import pandas as pd
from aws_aurora_dsql import create_table_and_load_df_to_aurora
from aws_bedrock import query_bedrock_model
from aws_s3 import read_parquet_files_from_s3_prefix
from ml_training_data_building import convert_news_topic_matching_output_to_df
from dotenv import load_dotenv
import os
from news_ingestion_pipeline import fetch_news_thenewsapi


def match_news_to_topics(news_data: pd.DataFrame) -> pd.DataFrame:
    news_data = (
        news_data
        .drop_duplicates(subset=["uuid"])
        .reset_index(drop=True)
    )

    all_result_dfs = []
    system_prompt = Path("src/prompt/text_to_topic_system_prompt.txt").read_text(encoding="utf-8")

    for _, row in news_data.iterrows():
        uuid = row["uuid"]
        title = row["title"]
        published_at = row["published_at"]
        source = row["source"]

        if pd.isna(title) or not str(title).strip():
            continue
        else:
            user_prompt = f"Title: {str(title).strip()}"

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

        topic_df = convert_news_topic_matching_output_to_df(
            output_text=output_text,
            uuid=uuid,
            title=title,
            published_at=published_at,
            source=source,
        )

        if not topic_df.empty:
            all_result_dfs.append(topic_df)

    if not all_result_dfs:
        return pd.DataFrame(
            columns=["uuid", "title", "published_at", "source", "topic", "confidence_score", "reason",]
        )

    return pd.concat(all_result_dfs, ignore_index=True)



if __name__ == "__main__":
    load_dotenv()
    
    news_df = read_parquet_files_from_s3_prefix(
            bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
            prefix=os.getenv("AWS_S3_OBJECT_KEY_DAILY_NEWS")
    )
    news_df = news_df[news_df["published_at"] >= "2026-04-13"]
    print(news_df)
    
    matching_df = match_news_to_topics(news_df)
    print(matching_df)
    
    create_table_and_load_df_to_aurora(
        df=matching_df,
        host=os.getenv("AWS_AURORA_DB_HOST"),
        database="postgres",
        schema_name="training_data",
        table_name="news_topic_matching",
        create_table=False,
    )
