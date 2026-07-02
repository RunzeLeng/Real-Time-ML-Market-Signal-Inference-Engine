from src.infrastructure.aws_aurora_dsql import AuroraDsqlClient
from src.infrastructure.aws_aurora_pgvector import AuroraPgVectorClient
from src.infrastructure.aws_s3 import S3StorageService
from src.retrieval_augmented_generation.rag_orchestration import RagOrchestration


if __name__ == "__main__":
    aurora_dsql_client = AuroraDsqlClient()
    aurora_pgvector_client = AuroraPgVectorClient()
    s3_service = S3StorageService()

    rag_orchestration = RagOrchestration(
        s3_service=s3_service,
        aurora_dsql_client=aurora_dsql_client,
        aurora_pgvector_client=aurora_pgvector_client,
    )

    news_topic_matching_query = """
    SELECT * FROM training_data.news_topic_matching
    """

    match_df = aurora_dsql_client.dsql_execute_sql(
        news_topic_matching_query
    )
    print(match_df)

    news_df = s3_service.read_daily_news()
    print(news_df)

    df_for_chunking = rag_orchestration.build_articles(
        news_topic_matching=match_df,
        source_df=news_df,
    )
    print(df_for_chunking)

    aurora_dsql_client.create_table_and_load_df_to_aurora(
        df=df_for_chunking,
        schema_name="training_data",
        table_name="df_for_chunking",
        create_table=True,
    )

    df_for_chunking_query = """
    SELECT * FROM training_data.df_for_chunking
    """

    df_for_chunking = aurora_dsql_client.dsql_execute_sql(
        df_for_chunking_query
    )
    print(df_for_chunking)

    df_for_chunking = rag_orchestration.article_prep_service.convert_topics_column_to_list(
        df_for_chunking,
    )
    print(df_for_chunking)

    chunked_df = rag_orchestration.build_chunk_df(
        df_for_chunking=df_for_chunking,
        uuid_col="uuid",
        text_col="full_text",
        add_metadata_header=True,
    )
    print(chunked_df)

    aurora_pgvector_client.load_df_to_table(
        df=chunked_df,
        table_name="chunked_df",
        schema_name="rag",
        create_table=True,
    )

    chunked_df_query = """
    SELECT * FROM rag.chunked_df
    """

    chunked_df = aurora_pgvector_client.execute_query(
        query=chunked_df_query,
    )
    print(chunked_df)

    embedding_df = rag_orchestration.build_embedding_vector(
        chunked_df=chunked_df,
        text_col="chunk_text",
        embedding_col="embedding_vector",
    )
    print(embedding_df)

    embedding_df.to_parquet(
        "output_embedding_df.parquet",
        index=False,
    )

    rag_orchestration.load_embedding_df_to_pgvector(
        embedding_df=embedding_df,
        schema_name="rag",
        table_name="embedding_df",
        create_table=True,
    )

    embedding_df_query = """
    SELECT * FROM rag.embedding_df
    """

    embedding_df = aurora_pgvector_client.execute_query(
        query=embedding_df_query,
    )
    print(embedding_df)
