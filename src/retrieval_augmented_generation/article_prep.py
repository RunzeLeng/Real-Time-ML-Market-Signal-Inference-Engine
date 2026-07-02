import pandas as pd



class ArticlePrepService:

    def prepare_articles_for_chunking(
        self,
        news_topic_matching: pd.DataFrame,
        source_df: pd.DataFrame,
    ) -> pd.DataFrame:

        uuid_level_df = (
            news_topic_matching
            .groupby(["uuid", "title", "published_at", "source"], as_index=False)
            .agg(
                topics=("topic", lambda values: list(dict.fromkeys(values)))
            )
        )

        merged_df = uuid_level_df.merge(
            source_df[["uuid", "full_text"]],
            on="uuid",
            how="left",
        )

        unique_uuid_count = news_topic_matching["uuid"].nunique()
        merged_row_count = len(merged_df)

        if unique_uuid_count != merged_row_count:
            raise ValueError(
                f"Merge row count mismatch: merged_df has {merged_row_count} rows, "
                f"but news_topic_matching has {unique_uuid_count} unique UUIDs."
            )

        return merged_df



    def convert_topics_column_to_list(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def parse_topics(value) -> list[str]:
            if isinstance(value, list):
                return value

            value = str(value).strip()

            if value.startswith("{") and value.endswith("}"):
                value = value[1:-1]

            return [
                topic.strip() for topic in value.split(",")
            ]

        df["topics"] = df["topics"].apply(parse_topics)

        return df
