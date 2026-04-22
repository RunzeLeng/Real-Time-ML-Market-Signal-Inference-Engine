from pathlib import Path
import pandas as pd
from aws_aurora_dsql import create_table_and_load_df_to_aurora, dsql_execute_sql
from aws_aurora_pgvector import execute_aurora_pgvector_query, load_df_to_aurora_pgvector_table
from aws_bedrock import query_bedrock_model
from aws_s3 import read_parquet_files_from_s3_prefix
from ml_training_data_building import convert_news_topic_matching_output_to_df
from dotenv import load_dotenv
import os
import re
import secrets
import string
import json
import boto3
import time
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError


def prepare_articles_for_chunking(
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



def count_tokens_approx(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0

    return len(re.findall(r"\w+|[^\w\s]", text))



def split_paragraphs(
    text: str,
    sentence_per_newline_threshold: float = 1.55,
) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    raw_text = text.strip()

    non_empty_lines = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
    ]

    if not non_empty_lines:
        return []

    total_newline_blocks = len(non_empty_lines)

    normalized_text = re.sub(r"\s+", " ", raw_text)
    total_sentences = len(split_sentences_in_paragraph(normalized_text))

    sentence_per_line_ratio = (
        total_sentences / total_newline_blocks
        if total_newline_blocks > 0
        else total_sentences
    )

    sentence_per_line_format = sentence_per_line_ratio <= sentence_per_newline_threshold

    if sentence_per_line_format:
        merged_text = re.sub(r"\s+", " ", raw_text).strip()
        return [merged_text]

    return [
        re.sub(r"\s+", " ", line).strip()
        for line in non_empty_lines
    ]



def split_sentences_in_paragraph(paragraph: str) -> list[str]:
    if not isinstance(paragraph, str) or not paragraph.strip():
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9\"'])",
        paragraph.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]



def sentence_starts_with_transition(sentence: str) -> bool:
    transition_patterns = [
        # contrast / opposition
        "however",
        "but",
        "yet",
        "still",
        "nevertheless",
        "nonetheless",
        "even so",
        "even then",
        "despite this",
        "despite that",
        "in contrast",
        "by contrast",
        "on the other hand",
        "on the contrary",
        "conversely",
        "instead",
        "rather",
        "alternatively",

        # continuation / addition
        "meanwhile",
        "also",
        "furthermore",
        "moreover",
        "in addition",
        "additionally",
        "besides",
        "what is more",
        "more broadly",
        "more importantly",
        "at the same time",
        "similarly",
        "likewise",

        # cause / consequence
        "therefore",
        "thus",
        "as a result",
        "consequently",
        "accordingly",
        "for this reason",
        "because of this",
        "because of that",
        "this means",
        "that means",
        "this suggests",
        "this indicates",
        "this reflects",
        "this underscores",
        "this highlights",
        "this marks",
        "this signals",

        # emphasis / clarification
        "indeed",
        "in fact",
        "notably",
        "significantly",
        "importantly",
        "crucially",
        "to be clear",
        "in other words",
        "put differently",
        "in practical terms",
        "in this context",
        "under these conditions",

        # examples / evidence
        "for example",
        "for instance",
        "such as",
        "including",
        "according to",
        "citing",
        "as evidence",
        "for comparison",

        # time / sequence
        "then",
        "next",
        "later",
        "earlier",
        "previously",
        "subsequently",
        "afterward",
        "afterwards",
        "since then",
        "from there",
        "over time",
        "in recent years",
        "in recent months",
        "in recent weeks",
        "in the meantime",

        # topic shift / section movement
        "separately",
        "elsewhere",
        "in another development",
        "on another front",
        "turning to",
        "regarding",
        "as for",
        "with respect to",
        "when it comes to",
        "in terms of",

        # conclusion / summary
        "overall",
        "ultimately",
        "finally",
        "in conclusion",
        "to conclude",
        "in summary",
        "to summarize",
        "the result is",
        "the outcome is",
    ]

    sentence_lower = sentence.lower().strip()

    return any(
        sentence_lower.startswith(pattern)
        for pattern in transition_patterns
    )



def build_sentence_units(text: str) -> list[dict]:
    units = []

    paragraphs = split_paragraphs(text)

    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = split_sentences_in_paragraph(paragraph)

        for sentence_index, sentence in enumerate(sentences):
            units.append({
                "text": sentence,
                "tokens": count_tokens_approx(sentence),
                "paragraph_index": paragraph_index,
                "sentence_index": sentence_index,
                "is_paragraph_end": sentence_index == len(sentences) - 1,
                "starts_with_transition": sentence_starts_with_transition(sentence),
            })

    return units



def get_overlap_units(previous_chunk_units: list[dict], overlap_tokens: int) -> list[dict]:
    if not previous_chunk_units:
        return []

    overlap_units = []
    token_count = 0

    for unit in reversed(previous_chunk_units):
        overlap_units.insert(0, unit)
        token_count += unit["tokens"]

        if token_count >= overlap_tokens:
            break

    return overlap_units



def expand_final_chunk_with_previous_context(
    current_units: list[dict],
    previous_chunk_units: list[dict],
    min_chunk_tokens: int,
) -> list[dict]:
    current_tokens = sum(unit["tokens"] for unit in current_units)

    if current_tokens >= min_chunk_tokens:
        return current_units

    current_texts = {unit["text"] for unit in current_units}
    expanded_units = current_units.copy()

    for unit in reversed(previous_chunk_units):
        if unit["text"] in current_texts:
            continue

        expanded_units.insert(0, unit)
        current_texts.add(unit["text"])
        current_tokens += unit["tokens"]

        if current_tokens >= min_chunk_tokens:
            break

    return expanded_units



def chunk_text_semantic_sentence_aware(
    text: str,
    min_chunk_tokens: int = 250,
    max_chunk_tokens: int = 350,
    overlap_tokens: int = 50,
) -> list[dict]:
    units = build_sentence_units(text)

    if not units:
        return []

    chunks = []
    previous_chunk_units = []
    i = 0

    while i < len(units):
        current_units = get_overlap_units(
            previous_chunk_units,
            overlap_tokens,
        )

        current_tokens = sum(unit["tokens"] for unit in current_units)
        added_new_sentence = False
        stop_reason = "end_of_document"

        while i < len(units):
            next_unit = units[i]

            if added_new_sentence and current_tokens + next_unit["tokens"] > max_chunk_tokens:
                stop_reason = "max_token_stop"
                break

            current_units.append(next_unit)
            current_tokens += next_unit["tokens"]
            added_new_sentence = True
            i += 1

            next_sentence = units[i] if i < len(units) else None

            reached_min_size = current_tokens >= min_chunk_tokens
            at_paragraph_boundary = next_unit["is_paragraph_end"]
            next_starts_transition = (
                next_sentence is not None
                and next_sentence["starts_with_transition"]
            )
            next_would_exceed_max = (
                next_sentence is not None
                and current_tokens + next_sentence["tokens"] > max_chunk_tokens
            )

            if i >= len(units):
                stop_reason = "end_of_document"
                break

            if reached_min_size and at_paragraph_boundary:
                stop_reason = "paragraph_boundary"
                break

            if reached_min_size and next_starts_transition:
                stop_reason = "transition_word"
                break

            if next_would_exceed_max:
                stop_reason = "max_token_stop"
                break

        if i >= len(units):
            current_units = expand_final_chunk_with_previous_context(
                current_units=current_units,
                previous_chunk_units=previous_chunk_units,
                min_chunk_tokens=min_chunk_tokens,
            )

            stop_reason = "end_of_document"

        chunk_text = " ".join(unit["text"] for unit in current_units).strip()

        if chunk_text:
            chunks.append({
                "chunk_text": chunk_text,
                "chunk_token_count": count_tokens_approx(chunk_text),
                "stop_reason": stop_reason,
            })

        previous_chunk_units = current_units

    return chunks



def chunk_news_for_embedding(
    df: pd.DataFrame,
    uuid_col: str = "uuid",
    text_col: str = "full_text",
    min_chunk_tokens: int = 200,
    max_chunk_tokens: int = 380,
    overlap_tokens: int = 50,
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        uuid = row[uuid_col]
        text = row[text_col]

        if pd.isna(text) or not str(text).strip():
            continue

        chunks = chunk_text_semantic_sentence_aware(
            text=str(text),
            min_chunk_tokens=min_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )

        for chunk_index, chunk in enumerate(chunks, start=1):
            rows.append({
                "uuid": uuid,
                "chunk_id": chunk_index,
                "chunk_text": chunk["chunk_text"],
                "chunk_token_count": chunk["chunk_token_count"],
                "stop_reason": chunk["stop_reason"],
            })

    return pd.DataFrame(
        rows,
        columns=[
            "uuid",
            "chunk_id",
            "chunk_text",
            "chunk_token_count",
            "stop_reason",
        ],
    )



def add_metadata_header_to_chunk_text(
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



def convert_topics_column_to_list(df: pd.DataFrame) -> pd.DataFrame:
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



def embed_text_with_titan_v2(
    text: str,
    region_name: str = "us-east-1",
    dimensions: int = 1024,
    normalize: bool = True,
) -> list[float]:
    client = boto3.client("bedrock-runtime", region_name=region_name)

    body = {
        "inputText": text,
        "dimensions": dimensions,
        "normalize": normalize,
        "embeddingTypes": ["float"],
    }

    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response["body"].read())

    return response_body["embedding"]



def add_titan_embeddings_to_df(
    df: pd.DataFrame,
    text_col: str = "chunk_text",
    embedding_col: str = "embedding_vector",
    max_workers: int = 3,
    max_attempts: int = 5,
    region_name: str = "us-east-1",
    dimensions: int = 1024,
    normalize: bool = True,
) -> pd.DataFrame:

    df = df.copy().reset_index(drop=True)
    results = [None] * len(df)

    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    df["_chunk_text_hash"] = df[text_col].fillna("").astype(str).apply(text_hash)

    def embed_one(index: int, text: str, expected_hash: str):
        if not isinstance(text, str) or not text.strip():
            return index, expected_hash, None

        actual_hash = text_hash(text)

        if actual_hash != expected_hash:
            raise ValueError(f"Text hash mismatch before embedding at index {index}")

        for attempt in range(max_attempts):
            try:
                embedding = embed_text_with_titan_v2(
                    text=text,
                    region_name=region_name,
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



if __name__ == "__main__":
    load_dotenv()
    
    # sql_query_2 = """
    # SELECT * FROM training_data.news_topic_matching
    # """
    
    # match_df = dsql_execute_sql(
    #     host=os.getenv("AWS_AURORA_DB_HOST"),
    #     database="postgres",
    #     sql=sql_query_2,
    #     user="admin",
    #     region="us-east-1",
    #     profile="default",
    # )
    # print(match_df)
    
    # news_df = read_parquet_files_from_s3_prefix(
    #     bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
    #     prefix=os.getenv("AWS_S3_OBJECT_KEY_DAILY_NEWS"),
    # )
    
    # print(news_df)
    
    # df_for_chunking = prepare_articles_for_chunking(
    #     news_topic_matching=match_df,
    #     source_df=news_df,
    # )
    
    # print(df_for_chunking)
    
    # create_table_and_load_df_to_aurora(
    # df=df_for_chunking,
    # host=os.getenv("AWS_AURORA_DB_HOST"),
    # database="postgres",
    # schema_name="training_data",
    # table_name="df_for_chunking",
    # create_table=True,
    # )
    
    # sql_query_2 = """
    # SELECT * FROM training_data.df_for_chunking
    # """
    
    # df_for_chunking = dsql_execute_sql(
    #     host=os.getenv("AWS_AURORA_DB_HOST"),
    #     database="postgres",
    #     sql=sql_query_2,
    #     user="admin",
    #     region="us-east-1",
    #     profile="default",
    # )
    # print(df_for_chunking)
    
    # df_for_chunking = convert_topics_column_to_list(df_for_chunking)
    # print(df_for_chunking)
    
    # chunked_df = chunk_news_for_embedding(
    #     df=df_for_chunking,
    #     uuid_col="uuid",
    #     text_col="full_text",
    #     min_chunk_tokens=200,
    #     max_chunk_tokens=380,
    #     overlap_tokens=50,
    # )
    # print(chunked_df)
    
    # chunked_df = add_metadata_header_to_chunk_text(
    #     chunked_df=chunked_df,
    #     df_for_chunking=df_for_chunking,
    # )
    # print(chunked_df)
    
    # load_df_to_aurora_pgvector_table(
    #     df=chunked_df,
    #     schema_name="rag",
    #     table_name="chunked_df",
    #     create_table=True,
    # )
    
    # sql_query = """
    #     SELECT * FROM rag.chunked_df
    # """
    
    # chunked_df = execute_aurora_pgvector_query(
    #     query=sql_query,
    # )
    # print(chunked_df)
    
    # embedding_df = add_titan_embeddings_to_df(
    #     df=chunked_df,
    #     text_col="chunk_text",
    #     embedding_col="embedding_vector",
    #     max_workers=3,
    #     region_name="us-east-1",
    #     dimensions=1024,
    #     normalize=True,
    # )
    # print(embedding_df)
    
    # embedding_df.to_parquet(
    #     "output_embedding_df.parquet",
    #     index=False,
    # )
    
    # load_df_to_aurora_pgvector_table(
    #     df=embedding_df,
    #     schema_name="rag",
    #     table_name="embedding_df",
    #     create_table=True,
    # )
    
    sql_query = """
        SELECT * FROM rag.embedding_df
    """
    
    embedding_df = execute_aurora_pgvector_query(
        query=sql_query,
    )
    print(embedding_df)