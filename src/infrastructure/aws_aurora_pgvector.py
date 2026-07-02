import boto3
import pandas as pd
import psycopg
from psycopg import sql
from src.common.config import AuroraPgVectorConfig, config


class AuroraPgVectorClient:
    
    def __init__(
        self,
        aurora_pgvector_config: AuroraPgVectorConfig | None = None,
    ) -> None:
        self.config = aurora_pgvector_config or config.aurora_pgvector
        self.rds_client = boto3.client("rds", region_name=self.config.region)



    def get_iam_token(self) -> str:
        return self.rds_client.generate_db_auth_token(
            DBHostname=self.config.host,
            Port=self.config.port,
            DBUsername=self.config.user,
            Region=self.config.region,
        )



    def get_connection(self):
        return psycopg.connect(
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.dbname,
            user=self.config.user,
            password=self.get_iam_token(),
            sslmode="require",
        )



    def load_df_to_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema_name: str | None = None,
        create_table: bool = False,
    ) -> None:
        schema_name = schema_name or self.config.schema
        df_to_load = df.copy()

        for col in df_to_load.columns:
            df_to_load[col] = df_to_load[col].apply(
                lambda value, col=col: self._normalize_value(value, col)
            )

        column_defs = [
            sql.SQL("{} {}").format(
                sql.Identifier(col),
                sql.SQL(self._infer_sql_type(df[col], col)),
            )
            for col in df.columns
        ]

        create_table_sql = sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
            sql.Identifier(schema_name),
            sql.Identifier(table_name),
            sql.SQL(", ").join(column_defs),
        )

        insert_sql = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
            sql.Identifier(schema_name),
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.Identifier(col) for col in df_to_load.columns),
            sql.SQL(", ").join(sql.Placeholder() for _ in df_to_load.columns),
        )

        values = [
            tuple(row)
            for row in df_to_load.itertuples(index=False, name=None)
        ]

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                if create_table:
                    cur.execute(create_table_sql)

                if values:
                    cur.executemany(insert_sql, values)

            conn.commit()



    def execute_query(self, query: str) -> pd.DataFrame | None:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)

                if cur.description is None:
                    conn.commit()
                    return None

                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]

            conn.commit()

        return pd.DataFrame(rows, columns=columns)



    def _infer_sql_type(self, series: pd.Series, column_name: str) -> str:
        if column_name == self.config.vector_column:
            return f"vector({self.config.vector_dimension})"

        if pd.api.types.is_integer_dtype(series):
            return "BIGINT"

        if pd.api.types.is_float_dtype(series):
            return "DOUBLE PRECISION"

        if pd.api.types.is_bool_dtype(series):
            return "BOOLEAN"

        if pd.api.types.is_datetime64_any_dtype(series):
            return "TIMESTAMP"

        non_null = series.dropna()

        if not non_null.empty and non_null.map(type).eq(list).all():
            return "TEXT[]"

        return "TEXT"



    def _normalize_value(self, value, column_name: str):
        if column_name == self.config.vector_column:
            if hasattr(value, "tolist"):
                value = value.tolist()

            if isinstance(value, list):
                return "[" + ",".join(str(x) for x in value) + "]"

        if isinstance(value, list):
            return value

        if pd.isna(value):
            return None

        return value
