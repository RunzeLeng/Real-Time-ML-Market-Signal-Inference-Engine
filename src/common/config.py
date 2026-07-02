# src/common/config.py

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------

def _get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name, default)

    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")

    return value


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None:
        return default

    return float(value)


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.lower() in {"1", "true", "yes", "y"}


def setup_logging(level: int = logging.INFO) -> None:
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )

# ---------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AuroraDsqlConfig:
    host: str | None
    database: str | None
    user: str | None
    region: str | None
    profile: str | None


@dataclass(frozen=True)
class AuroraPgVectorConfig:
    region: str
    host: str
    port: int
    dbname: str
    user: str
    schema: str
    vector_column: str
    vector_dimension: int


@dataclass(frozen=True)
class DynamoDBConfig:
    region: str
    processed_records_table: str
    published_records_table: str


@dataclass(frozen=True)
class S3Config:
    region: str
    bucket_name: str
    post_object_key: str | None
    etf_object_key: str | None
    group_etf_object_key: str | None
    single_etf_object_key: str | None
    daily_news_object_key: str | None


@dataclass(frozen=True)
class SnsConfig:
    region: str
    topic_arn: str | None


@dataclass(frozen=True)
class ApifyConfig:
    token: str | None
    default_actor_id: str
    backup_actor_id: str


@dataclass(frozen=True)
class ScrapeOpsConfig:
    api_key: str | None
    endpoint: str | None
    base_url: str | None


@dataclass(frozen=True)
class NewsConfig:
    gdelt_base_url: str
    news_api_token: str | None
    news_api_base_url: str
    news_api_search_fields: str
    news_api_limit: int
    news_api_max_pages: int
    news_api_categories: str
    news_api_exclude_categories: str
    news_api_language: str
    news_api_locale: str


@dataclass(frozen=True)
class TopicMemoryConfig:
    model_id: str
    region: str
    topic_matching_system_prompt_path: str
    topic_summary_system_prompt_path: str
    topic_matching_schema: str
    topic_matching_table: str
    topic_summary_schema: str
    topic_summary_table: str


@dataclass(frozen=True)
class PerformanceReviewConfig:
    default_metric_col: str
    secondary_metric_col: str | None
    reasonableness_col: str
    high_reasonableness_threshold: float
    lookback_days: int
    review_weekday: int


@dataclass(frozen=True)
class MarketDataConfig:
    alpaca_api_key: str | None
    alpaca_secret_key: str | None


@dataclass(frozen=True)
class RagConfig:
    titan_embedding_model_id: str
    titan_embedding_region: str
    titan_embedding_dimension: int
    titan_embedding_normalize: bool
    chunk_min_tokens: int
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    embedding_max_workers: int
    embedding_max_attempts: int
    schema: str
    chunk_table: str
    embedding_table: str


@dataclass(frozen=True)
class LargeLanguageModelConfig:
    region: str
    read_timeout: int
    connect_timeout: int
    max_retry_attempts: int


@dataclass(frozen=True)
class AppConfig:
    aurora_dsql: AuroraDsqlConfig
    aurora_pgvector: AuroraPgVectorConfig
    dynamodb: DynamoDBConfig
    s3: S3Config
    sns: SnsConfig
    apify: ApifyConfig
    scrapeops: ScrapeOpsConfig
    news: NewsConfig
    topic_memory: TopicMemoryConfig
    performance_review: PerformanceReviewConfig
    market_data: MarketDataConfig
    rag: RagConfig
    large_language_model: LargeLanguageModelConfig
    

# ---------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------

def load_config() -> AppConfig:

    return AppConfig(
        
        aurora_dsql=AuroraDsqlConfig(
            host=_get_env("AURORA_DSQL_HOST", required=True),
            database=_get_env("AURORA_DSQL_DATABASE", "postgres"),
            user=_get_env("AURORA_DSQL_USER", "admin"),
            region=_get_env("AURORA_DSQL_REGION", "us-east-1"),
            profile=_get_env("AURORA_DSQL_PROFILE", "default"),
        ),

        aurora_pgvector=AuroraPgVectorConfig(
            region=_get_env("AURORA_PGVECTOR_REGION", "us-east-1"),
            host=_get_env("AURORA_PGVECTOR_HOST", required=True),
            port=_get_int("AURORA_PGVECTOR_PORT", 5432),
            dbname=_get_env("AURORA_PGVECTOR_DBNAME", "postgres"),
            user=_get_env("AURORA_PGVECTOR_USER", "postgres"),
            schema=_get_env("AURORA_PGVECTOR_SCHEMA", "rag"),
            vector_column=_get_env("AURORA_PGVECTOR_VECTOR_COLUMN", "embedding_vector"),
            vector_dimension=_get_int("AURORA_PGVECTOR_VECTOR_DIMENSION", 1024),
        ),
        
        dynamodb=DynamoDBConfig(
            region=_get_env("DYNAMODB_REGION", "us-east-1"),
            processed_records_table=_get_env("DYNAMODB_PROCESSED_RECORDS_TABLE", "processed_records"),
            published_records_table=_get_env("DYNAMODB_PUBLISHED_RECORDS_TABLE", "published_records"),
        ),
        
        s3=S3Config(
            region=_get_env("AWS_S3_REGION", "us-east-1"),
            bucket_name=_get_env("AWS_S3_BUCKET_NAME", required=True),
            post_object_key=_get_env("AWS_S3_OBJECT_KEY_POST"),
            etf_object_key=_get_env("AWS_S3_OBJECT_KEY_ETF"),
            group_etf_object_key=_get_env("AWS_S3_OBJECT_KEY_GROUP_ETF"),
            single_etf_object_key=_get_env("AWS_S3_OBJECT_KEY_SINGLE_ETF"),
            daily_news_object_key=_get_env("AWS_S3_OBJECT_KEY_DAILY_NEWS"),
        ),

        sns=SnsConfig(
            region=_get_env("AWS_SNS_REGION", "us-east-1"),
            topic_arn=_get_env("AWS_SNS_TOPIC_ARN"),
        ),

        apify=ApifyConfig(
            token=_get_env("APIFY_TOKEN"),
            default_actor_id=_get_env("APIFY_DEFAULT_ACTOR_ID", "GsRHwiTFlQB8bh2yf"),
            backup_actor_id=_get_env("APIFY_BACKUP_ACTOR_ID", "sTDLfdZAmte0aYlxg"),
        ),

        scrapeops=ScrapeOpsConfig(
            api_key=_get_env("SCRAPEOPS_API_KEY"),
            endpoint=_get_env("SCRAPEOPS_ENDPOINT"),
            base_url=_get_env("SCRAPEOPS_BASE_URL"),
        ),

        news=NewsConfig(
            gdelt_base_url=_get_env("GDELT_BASE_URL", "https://api.gdeltproject.org/api/v2/doc/doc"),
            news_api_token=_get_env("NEWS_API_TOKEN"),
            news_api_base_url=_get_env("NEWS_API_BASE_URL", "https://api.thenewsapi.com/v1/news/all"),
            news_api_search_fields=_get_env("NEWS_API_SEARCH_FIELDS", "title,main_text"),
            news_api_limit=_get_int("NEWS_API_LIMIT", 25),
            news_api_max_pages=_get_int("NEWS_API_MAX_PAGES", 20),
            news_api_categories=_get_env("NEWS_API_CATEGORIES", "business,politics"),
            news_api_exclude_categories=_get_env("NEWS_API_EXCLUDE_CATEGORIES", "travel,food,entertainment,health,sports"),
            news_api_language=_get_env("NEWS_API_LANGUAGE", "en"),
            news_api_locale=_get_env("NEWS_API_LOCALE", "us"),
        ),

        topic_memory=TopicMemoryConfig(
            model_id=_get_env("TOPIC_MEMORY_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
            region=_get_env("TOPIC_MEMORY_REGION", "us-east-1"),
            topic_matching_system_prompt_path=_get_env("TOPIC_MATCHING_SYSTEM_PROMPT_PATH", "src/prompt/text_to_topic_system_prompt.txt"),
            topic_summary_system_prompt_path=_get_env("TOPIC_SUMMARY_SYSTEM_PROMPT_PATH", "src/prompt/topic_summary_system_prompt.txt"),
            topic_matching_schema=_get_env("TOPIC_MATCHING_SCHEMA", "training_data"),
            topic_matching_table=_get_env("TOPIC_MATCHING_TABLE", "news_topic_matching"),
            topic_summary_schema=_get_env("TOPIC_SUMMARY_SCHEMA", "training_data"),
            topic_summary_table=_get_env("TOPIC_SUMMARY_TABLE", "topic_summary"),
        ),

        performance_review=PerformanceReviewConfig(
            default_metric_col=_get_env("PERFORMANCE_REVIEW_DEFAULT_METRIC_COL", "vwap_pct_change_30m"),
            secondary_metric_col=_get_env("PERFORMANCE_REVIEW_SECONDARY_METRIC_COL", "vwap_pct_change_3h"),
            reasonableness_col=_get_env("PERFORMANCE_REVIEW_REASONABLENESS_COL", "reasonableness_score"),
            high_reasonableness_threshold=_get_float("PERFORMANCE_REVIEW_HIGH_REASONABLENESS_THRESHOLD", 0.5),
            lookback_days=_get_int("PERFORMANCE_REVIEW_LOOKBACK_DAYS", 7),
            review_weekday=_get_int("PERFORMANCE_REVIEW_REVIEW_WEEKDAY", 0),
        ),

        market_data=MarketDataConfig(
            alpaca_api_key=_get_env("ALPACA_API_KEY"),
            alpaca_secret_key=_get_env("ALPACA_SECRET_KEY"),
        ),

        rag=RagConfig(
            titan_embedding_model_id=_get_env("RAG_TITAN_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            titan_embedding_region=_get_env("RAG_TITAN_EMBEDDING_REGION", "us-east-1"),
            titan_embedding_dimension=_get_int("RAG_TITAN_EMBEDDING_DIMENSION", 1024),
            titan_embedding_normalize=_get_bool("RAG_TITAN_EMBEDDING_NORMALIZE", True),
            chunk_min_tokens=_get_int("RAG_CHUNK_MIN_TOKENS", 200),
            chunk_max_tokens=_get_int("RAG_CHUNK_MAX_TOKENS", 380),
            chunk_overlap_tokens=_get_int("RAG_CHUNK_OVERLAP_TOKENS", 50),
            embedding_max_workers=_get_int("RAG_EMBEDDING_MAX_WORKERS", 3),
            embedding_max_attempts=_get_int("RAG_EMBEDDING_MAX_ATTEMPTS", 5),
            schema=_get_env("RAG_SCHEMA", "rag"),
            chunk_table=_get_env("RAG_CHUNK_TABLE", "chunked_df"),
            embedding_table=_get_env("RAG_EMBEDDING_TABLE", "embedding_df"),
        ),

        large_language_model=LargeLanguageModelConfig(
            region=_get_env("LLM_REGION", "us-east-1"),
            read_timeout=_get_int("LLM_READ_TIMEOUT", 300),
            connect_timeout=_get_int("LLM_CONNECT_TIMEOUT", 60),
            max_retry_attempts=_get_int("LLM_MAX_RETRY_ATTEMPTS", 3),
        ),

    )

config = load_config()
