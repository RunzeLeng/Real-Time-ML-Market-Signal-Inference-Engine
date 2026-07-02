from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import random
import time
import boto3
import pandas as pd
from botocore.exceptions import ClientError
from src.common.config import RagConfig, config



class TitanEmbeddingService:

    def __init__(self, rag_config: RagConfig | None = None) -> None:
        self.config = rag_config or config.rag
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=self.config.titan_embedding_region,
        )



    def add_metadata_header_to_chunked_text(
        self,
        chunked_df: pd.DataFrame,
        df_for_chunking: pd.DataFrame,
    ) -> pd.DataFrame:
        
        metadata_df = df_for_chunking.copy()
        merged_df = chunked_df.merge(
            metadata_df,
            on="uuid",
            how="inner",
        )

        def format_topics(topics) -> str:
            return ", ".join(str(topic) for topic in topics)

        merged_df["chunk_text"] = merged_df.apply(
            lambda row: (
                f"Title: {row['title']}\n"
                f"Published at: {row['published_at']} | Source: {row['source']}\n"
                f"Topics: {format_topics(row['topics'])}\n"
                f"{row['chunk_text']}"
            ),
            axis=1,
        )

        if len(merged_df) != len(chunked_df) or merged_df["uuid"].nunique() != chunked_df["uuid"].nunique():
            raise ValueError(
                f"Metadata merge error: merged_df has {len(merged_df)} rows and {merged_df['uuid'].nunique()} unique UUIDs, "
                f"but chunked_df has {len(chunked_df)} rows and {chunked_df['uuid'].nunique()} unique UUIDs."
            )

        return merged_df.drop(columns=["full_text"])



    def embed_text(
        self,
        text: str,
        dimensions: int | None = None,
        normalize: bool | None = None,
    ) -> list[float]:
        
        dimensions = dimensions or self.config.titan_embedding_dimension
        normalize = self.config.titan_embedding_normalize if normalize is None else normalize

        body = {
            "inputText": text,
            "dimensions": dimensions,
            "normalize": normalize,
            "embeddingTypes": ["float"],
        }

        response = self.client.invoke_model(
            modelId=self.config.titan_embedding_model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response["body"].read())

        return response_body["embedding"]



    def add_embeddings_to_df(
        self,
        df: pd.DataFrame,
        text_col: str = "chunk_text",
        embedding_col: str = "embedding_vector",
        max_workers: int | None = None,
        max_attempts: int | None = None,
        dimensions: int | None = None,
        normalize: bool | None = None,
    ) -> pd.DataFrame:
        
        max_workers = max_workers or self.config.embedding_max_workers
        max_attempts = max_attempts or self.config.embedding_max_attempts

        df = df.copy().reset_index(drop=True)
        results = [None] * len(df)

        df["_chunk_text_hash"] = df[text_col].fillna("").astype(str).apply(self.text_hash)

        def embed_one(index: int, text: str, expected_hash: str):
            if not isinstance(text, str) or not text.strip():
                return index, expected_hash, None

            actual_hash = self.text_hash(text)

            if actual_hash != expected_hash:
                raise ValueError(f"Text hash mismatch before embedding at index {index}")

            for attempt in range(max_attempts):
                try:
                    embedding = self.embed_text(
                        text=text,
                        dimensions=dimensions,
                        normalize=normalize,
                    )

                    return index, expected_hash, embedding

                except ClientError as error:
                    error_code = error.response["Error"]["Code"]

                    if error_code not in {"ThrottlingException", "TooManyRequestsException"}:
                        raise

                    if attempt == max_attempts - 1:
                        raise

                    sleep_seconds = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(sleep_seconds)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    embed_one,
                    index,
                    row[text_col],
                    row["_chunk_text_hash"],
                )
                for index, row in df.iterrows()
            ]

            for future in as_completed(futures):
                index, expected_hash, embedding = future.result()

                if df.loc[index, "_chunk_text_hash"] != expected_hash:
                    raise ValueError(f"Text hash mismatch after embedding at index {index}")

                results[index] = embedding
                print(f"Completed embedding for index {index + 1}/{len(df)}")

        df[embedding_col] = results
        df = df.drop(columns=["_chunk_text_hash"])

        return df



    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()