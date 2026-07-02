# src/infrastructure/aws_aurora_dsql.py
import psycopg
import aurora_dsql_psycopg as dsql
from psycopg import sql
import pandas as pd
from src.common.config import config



class AuroraDsqlClient:
    
    def __init__(self):
        self.config = config.aurora_dsql


    def _connection_params(self) -> dict:
        return {
            "host": self.config.host,
            "dbname": self.config.database,
            "user": self.config.user,
            "region": self.config.region,
        }


    def dsql_execute_sql(self, sql_query: str) -> pd.DataFrame | None:
        with dsql.connect(**self._connection_params()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)

                try:
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description]
                    return pd.DataFrame(rows, columns=columns)
                except psycopg.ProgrammingError:
                    return None


    def create_table_and_load_df_to_aurora(
        self,
        df: pd.DataFrame,
        schema_name: str,
        table_name: str,
        create_table: bool = False,
    ) -> None:
        df_to_load = df.copy()
        df_to_load = df_to_load.where(pd.notna(df_to_load), None)

        column_defs = [
            sql.SQL("{} {}").format(
                sql.Identifier(col),
                sql.SQL(self._infer_sql_type(df[col])),
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
            sql.SQL(", ").join(sql.Identifier(col) for col in df.columns),
            sql.SQL(", ").join(sql.Placeholder() for _ in df.columns),
        )

        values = [tuple(row) for row in df_to_load.itertuples(index=False, name=None)]

        with dsql.connect(**self._connection_params()) as conn:
            if create_table:
                with conn.cursor() as cur:
                    cur.execute(create_table_sql)
                conn.commit()

            if values:
                with conn.cursor() as cur:
                    cur.executemany(insert_sql, values)
                conn.commit()
    
    
    @staticmethod
    def _infer_sql_type(series: pd.Series) -> str:
        if pd.api.types.is_integer_dtype(series):
            return "BIGINT"
        if pd.api.types.is_float_dtype(series):
            return "DOUBLE PRECISION"
        if pd.api.types.is_bool_dtype(series):
            return "BOOLEAN"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "TIMESTAMP"
        return "TEXT"
    

