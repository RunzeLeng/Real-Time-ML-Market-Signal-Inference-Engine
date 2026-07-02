import pandas as pd



class PostProcessingService:
    

    def post_filtering(
        self,
        df: pd.DataFrame,
        num_posts: int = 5,
    ) -> pd.DataFrame:
        
        filtered_df = (
            df
            .assign(content=lambda d: d["content"].str.replace(r"<.*?>", "", regex=True).str.strip())
            .query("content != ''")
            .loc[lambda d: ~d["content"].str.startswith("https://truthsocial.com/")]
            .loc[lambda d: ~d["content"].str.startswith("https://www")]
            .loc[lambda d: ~d["content"].str.startswith("https://dailycaller")]
            .loc[lambda d: ~d["content"].str.startswith("RT")]
            .head(num_posts)
        )

        return filtered_df



    def post_filtering_for_multimodal(
        self,
        df: pd.DataFrame,
        num_posts: int = 5,
    ) -> pd.DataFrame:
        
        filtered_df = (
            df
            .assign(content=lambda d: d["content"].str.replace(r"<.*?>", "", regex=True).str.strip())
            .loc[
                lambda d: (
                    d["content"].ne("") | d["media"].apply(lambda value: bool(value))
                )
            ]
            .loc[lambda d: ~d["content"].str.startswith("https://truthsocial.com/")]
            .loc[lambda d: ~d["content"].str.startswith("RT")]
            .head(num_posts)
        )

        return filtered_df



    def post_formating(
        self,
        df: pd.DataFrame,
        column: str = "created_at",
    ) -> pd.DataFrame:
        
        formatted_df = df.copy()
        formatted_df[column] = pd.to_datetime(formatted_df[column], utc=True)
        formatted_df[column] = formatted_df[column].dt.floor("s").dt.tz_convert("US/Eastern")

        return formatted_df



    def filter_posts_by_date_and_content_length(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        min_content_length: int,
        date_column: str = "created_at",
        content_column: str = "content",
    ) -> pd.DataFrame:
        
        filtered_df = df.copy()
        filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors="coerce")
        
        start_date = pd.Timestamp(start_date, tz="US/Eastern")
        end_date = pd.Timestamp(end_date, tz="US/Eastern")

        filtered_df = filtered_df[
            (filtered_df[date_column] >= start_date)
            & (filtered_df[date_column] <= end_date)
        ]

        filtered_df = filtered_df[
            filtered_df[content_column].fillna("").str.len() > min_content_length
        ]

        return filtered_df



    def duplicate_posts_to_minute_boundaries(
        self,
        df: pd.DataFrame,
        datetime_column: str = "created_at",
        post_duplicate: bool = True,
    ) -> pd.DataFrame:
        
        base_df = df.copy()
        base_df[datetime_column] = pd.to_datetime(base_df[datetime_column], errors="coerce")
        base_df["created_at_seconds"] = base_df[datetime_column]

        first_df = base_df.copy()
        first_df[datetime_column] = (
            first_df[datetime_column]
            .dt.tz_convert("UTC")
            .dt.floor("min")
            .dt.tz_convert("US/Eastern")
        )

        second_df = base_df.copy()
        second_df[datetime_column] = (
            second_df[datetime_column]
            .dt.tz_convert("UTC")
            .dt.floor("min")
            .dt.tz_convert("US/Eastern")
            + pd.Timedelta(minutes=1)
        )

        df_rounding = base_df.copy()
        df_rounding[datetime_column] = (
            df_rounding[datetime_column]
            .dt.tz_convert("UTC")
            .dt.round("min")
            .dt.tz_convert("US/Eastern")
        )

        if post_duplicate:
            result_df = pd.concat([first_df, second_df], ignore_index=True)
        else:
            result_df = df_rounding

        return result_df



    def add_post_prefix_to_content(self, df: pd.DataFrame) -> pd.DataFrame:
        prefixed_df = df.copy()
        prefixed_df["content"] = "Post: " + prefixed_df["content"].astype(str)
        
        return prefixed_df



    def dedupe_posts(
        self,
        df: pd.DataFrame,
        processed_post_ids: set,
        id_col: str = "id",
    ) -> pd.DataFrame:
        
        filtered_df = df[~df[id_col].isin(processed_post_ids)].copy()
        return filtered_df.head(1)



    def build_user_prompt_from_post(self, df: pd.DataFrame) -> str:
        content = str(df.iloc[0]["content"]).strip()
        return f"Post: {content}"



    def add_id_to_processed_post_ids(
        self,
        df: pd.DataFrame,
        processed_post_ids: set[str],
    ) -> set[str]:
        if not df.empty:
            processed_post_ids.add(df.iloc[0]["id"])

        return processed_post_ids



    def extract_ids_and_contents(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        ids = df["id"].astype(str).tolist()
        contents = df["content"].astype(str).tolist()
        
        return ids, contents



    def deduplicate_and_remove_existing_ids(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
    ) -> pd.DataFrame:
        
        df1_dedup = df1.drop_duplicates(subset=["id"]).copy()
        existing_ids = set(df2["id"].dropna())
        result_df = df1_dedup[~df1_dedup["id"].isin(existing_ids)].copy()

        return result_df