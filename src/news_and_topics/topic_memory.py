from pathlib import Path
import pandas as pd
from src.common.config import TopicMemoryConfig, config
from src.infrastructure.aws_aurora_dsql import AuroraDsqlClient
from src.large_language_model.large_language_model import LLMService
from src.processing.json_processing import JSONProcessingService


class TopicMemoryService:
    
    def __init__(
        self,
        topic_memory_config: TopicMemoryConfig | None = None,
        aurora_client: AuroraDsqlClient | None = None,
        llm_service: LLMService | None = None,
        json_processing_service: JSONProcessingService | None = None,
    ) -> None:
        
        self.config = topic_memory_config or config.topic_memory
        self.aurora_client = aurora_client or AuroraDsqlClient()
        self.llm_service = llm_service or LLMService()
        self.json_processing_service = json_processing_service or JSONProcessingService()



    def match_news_to_topics(self, news_data: pd.DataFrame) -> pd.DataFrame:
        
        news_data = (
            news_data
            .drop_duplicates(subset=["uuid"])
            .reset_index(drop=True)
        )

        all_result_dfs = []
        system_prompt = Path(self.config.topic_matching_system_prompt_path).read_text(encoding="utf-8")

        for _, row in news_data.iterrows():
            uuid = row["uuid"]
            title = row["title"]
            published_at = row["published_at"]
            source = row["source"]

            if pd.isna(title) or not str(title).strip():
                continue
            else:
                user_prompt = f"Title: {str(title).strip()}"

            output_text = self.llm_service.query_model(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_id=self.config.model_id,
                region_name=self.config.region,
                temperature=0.2,
                max_tokens=600,
                top_p=0.95,
                top_k=250,
                system_prompt_caching=True,
            )

            topic_df = self.json_processing_service.news_topic_matching_output_to_df(
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
                columns=["uuid", "title", "published_at", "source", "topic", "confidence_score", "reason"]
            )

        return pd.concat(all_result_dfs, ignore_index=True)



    def match_post_to_topics(self, id: str, post: str) -> pd.DataFrame:
        
        system_prompt = Path(self.config.topic_matching_system_prompt_path).read_text(encoding="utf-8")
        user_prompt = f"Title: {str(post).strip()}"

        output_text = self.llm_service.query_model(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_id=self.config.model_id,
            region_name=self.config.region,
            temperature=0.2,
            max_tokens=600,
            top_p=0.95,
            top_k=250,
            system_prompt_caching=True,
        )

        post_topic_match_df = self.json_processing_service.post_topic_matching_output_to_df(
            output_text=output_text,
            id=id,
            post=post,
        )

        return post_topic_match_df



    def summarize_news_by_topic(
        self,
        news_df: pd.DataFrame,
        processing_date: str,
    ) -> pd.DataFrame:
        system_prompt = Path(self.config.topic_summary_system_prompt_path).read_text(encoding="utf-8")
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

            model_output = self.llm_service.query_model(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_id=self.config.model_id,
                region_name=self.config.region,
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

            topic_summary_df = self.json_processing_service.topic_summary_output_to_df(
                topic=topic,
                processing_date=processing_date,
                output_text=model_output,
            )

            all_result_dfs.append(topic_summary_df)

        return pd.concat(all_result_dfs, ignore_index=True)



    def save_news_topic_matching(
        self,
        matching_df: pd.DataFrame,
        create_table: bool = False,
    ) -> None:
        
        self.aurora_client.create_table_and_load_df_to_aurora(
            df=matching_df,
            schema_name=self.config.topic_matching_schema,
            table_name=self.config.topic_matching_table,
            create_table=create_table,
        )



    def save_topic_summary(
        self,
        summary_df: pd.DataFrame,
        create_table: bool = False,
    ) -> None:
        
        self.aurora_client.create_table_and_load_df_to_aurora(
            df=summary_df,
            schema_name=self.config.topic_summary_schema,
            table_name=self.config.topic_summary_table,
            create_table=create_table,
        )
