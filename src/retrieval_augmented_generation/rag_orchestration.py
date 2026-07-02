import pandas as pd
from src.common.config import RagConfig, config
from src.infrastructure.aws_aurora_dsql import AuroraDsqlClient
from src.infrastructure.aws_aurora_pgvector import AuroraPgVectorClient
from src.infrastructure.aws_s3 import S3StorageService
from src.retrieval_augmented_generation.article_prep import ArticlePrepService
from src.retrieval_augmented_generation.embedding_service import TitanEmbeddingService
from src.retrieval_augmented_generation.semantic_chunking import SemanticChunkingService



class RagOrchestration:

    def __init__(
        self,
        rag_config: RagConfig | None = None,
        article_prep_service: ArticlePrepService | None = None,
        semantic_chunking_service: SemanticChunkingService | None = None,
        titan_embedding_service: TitanEmbeddingService | None = None,
        
        s3_service: S3StorageService | None = None,
        aurora_dsql_client: AuroraDsqlClient | None = None,
        aurora_pgvector_client: AuroraPgVectorClient | None = None,
    ) -> None:
        
        self.config = rag_config or config.rag
        self.article_prep_service = article_prep_service or ArticlePrepService()
        self.semantic_chunking_service = semantic_chunking_service or SemanticChunkingService(self.config)
        self.titan_embedding_service = titan_embedding_service or TitanEmbeddingService(self.config)
        
        self.s3_service = s3_service or S3StorageService()
        self.aurora_dsql_client = aurora_dsql_client or AuroraDsqlClient()
        self.aurora_pgvector_client = aurora_pgvector_client or AuroraPgVectorClient()



    def build_articles(
        self,
        news_topic_matching: pd.DataFrame,
        source_df: pd.DataFrame,
    ) -> pd.DataFrame:
        
        return self.article_prep_service.prepare_articles_for_chunking(
            news_topic_matching=news_topic_matching,
            source_df=source_df,
        )



    def build_chunk_df(
        self,
        df_for_chunking: pd.DataFrame,
        uuid_col: str = "uuid",
        text_col: str = "full_text",
        add_metadata_header: bool = True,
    ) -> pd.DataFrame:
        
        chunked_df = self.semantic_chunking_service.chunk_news_for_embedding(
            df=df_for_chunking,
            uuid_col=uuid_col,
            text_col=text_col,
        )

        if add_metadata_header:
            chunked_df = self.titan_embedding_service.add_metadata_header_to_chunked_text(
                chunked_df=chunked_df,
                df_for_chunking=df_for_chunking,
            )

        return chunked_df



    def build_embedding_vector(
        self,
        chunked_df: pd.DataFrame,
        text_col: str = "chunk_text",
        embedding_col: str = "embedding_vector",
    ) -> pd.DataFrame:
        
        return self.titan_embedding_service.add_embeddings_to_df(
            df=chunked_df,
            text_col=text_col,
            embedding_col=embedding_col,
        )



    def load_embedding_df_to_pgvector(
        self,
        embedding_df: pd.DataFrame,
        schema_name: str | None = None,
        table_name: str | None = None,
        create_table: bool = False,
    ) -> None:
        
        self.aurora_pgvector_client.load_df_to_table(
            df=embedding_df,
            schema_name=schema_name or self.config.schema,
            table_name=table_name or self.config.embedding_table,
            create_table=create_table,
        )
