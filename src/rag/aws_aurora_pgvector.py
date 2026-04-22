import os
import boto3
from dotenv import load_dotenv
import psycopg
import pandas as pd
from psycopg import sql


def get_iam_token() -> str:
    load_dotenv()
    
    rds = boto3.client("rds", region_name=os.getenv("AWS_AURORA_SERVERLESS_REGION"))
    return rds.generate_db_auth_token(
        DBHostname=os.getenv("AWS_AURORA_SERVERLESS_HOST"),
        Port=os.getenv("AWS_AURORA_SERVERLESS_PORT"),
        DBUsername=os.getenv("AWS_AURORA_SERVERLESS_USER"),
        Region=os.getenv("AWS_AURORA_SERVERLESS_REGION"),
    )



def get_connection():
    load_dotenv()
    
    conn = psycopg.connect(
        host=os.getenv("AWS_AURORA_SERVERLESS_HOST"),
        port=os.getenv("AWS_AURORA_SERVERLESS_PORT"),
        dbname=os.getenv("AWS_AURORA_SERVERLESS_DBNAME"),
        user=os.getenv("AWS_AURORA_SERVERLESS_USER"),
        password=get_iam_token(),
        sslmode="require",
    )
    return conn



def load_df_to_aurora_pgvector_table(
    df: pd.DataFrame,
    schema_name: str,
    table_name: str,
    create_table: bool = False,
) -> None:
    def infer_sql_type(
        series: pd.Series,
        column_name: str,
        vector_column: str = "embedding_vector",
        vector_dimension: int = 1024,
    ) -> str:
        if column_name == vector_column:
            return f"vector({vector_dimension})"
    
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

    df_to_load = df.copy()

    def normalize_value(value, column_name: str):
        if column_name == "embedding_vector":
            if hasattr(value, "tolist"):
                value = value.tolist()
            
            if isinstance(value, list):
                return "[" + ",".join(str(x) for x in value) + "]"
    
        if isinstance(value, list):
            return value

        if pd.isna(value):
            return None

        return value

    for col in df_to_load.columns:
        df_to_load[col] = df_to_load[col].apply(lambda x, col=col: normalize_value(x, col))

    column_defs = [
        sql.SQL("{} {}").format(
            sql.Identifier(col),
            sql.SQL(
                infer_sql_type(
                    series=df[col],
                    column_name=col,
                    vector_column="embedding_vector",
                    vector_dimension=1024,
                )
            ),
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
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            if create_table:
                cur.execute(create_table_sql)
            
            if values:
                cur.executemany(insert_sql, values)

        conn.commit()



def execute_aurora_pgvector_query(query: str) -> pd.DataFrame | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)

            if cur.description is None:
                conn.commit()
                return None

            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

        conn.commit()

    return pd.DataFrame(rows, columns=columns)
