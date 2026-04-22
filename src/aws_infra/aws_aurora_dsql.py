import psycopg
import aurora_dsql_psycopg as dsql
from psycopg import sql
import pandas as pd
from dotenv import load_dotenv
import os


def dsql_execute_sql(
    host: str,
    database: str,
    sql: str,
    user: str = "admin",
    region: str = "us-east-1",
    profile: str = "default",
):
    conn_params = {
        "host": host,
        "dbname": database,
        "user": user,
        "region": region,
    }

    with dsql.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)

            try:
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                return pd.DataFrame(rows, columns=columns)
            except psycopg.ProgrammingError:
                return None



def create_table_and_load_df_to_aurora(
    df: pd.DataFrame,
    host: str,
    database: str,
    schema_name: str,
    table_name: str,
    create_table: bool = False,
    user: str = "admin",
    region: str = "us-east-1",
    profile: str = "default",
) -> None:
    def infer_sql_type(series: pd.Series) -> str:
        if pd.api.types.is_integer_dtype(series):
            return "BIGINT"
        if pd.api.types.is_float_dtype(series):
            return "DOUBLE PRECISION"
        if pd.api.types.is_bool_dtype(series):
            return "BOOLEAN"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "TIMESTAMP"
        return "TEXT"

    conn_params = {
        "host": host,
        "dbname": database,
        "user": user,
        "region": region,
    }

    df_to_load = df.copy()
    df_to_load = df_to_load.where(pd.notna(df_to_load), None)

    column_defs = [
        sql.SQL("{} {}").format(
            sql.Identifier(col),
            sql.SQL(infer_sql_type(df[col])),
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

    with dsql.connect(**conn_params) as conn:
        if create_table:
            with conn.cursor() as cur:
                    cur.execute(create_table_sql)
            conn.commit()
        if values:
            with conn.cursor() as cur:
                    cur.executemany(insert_sql, values)
            conn.commit()



if __name__ == "__main__":
    load_dotenv()
    
    # sql_query_2 = """
    #     SELECT *, a.avg_valid_accuracy_high_confidence - b.avg_valid_accuracy_high_confidence AS accuracy_diff
    #     FROM training_output.selected_model_performance as a
    #     inner join training_output.model_performance as b
    #     on a.combo_id = b.combo_id
    #     and a.symbol = b.symbol
    #     order by a.symbol, a.combo_id;
    # """
    
    # sql_query_2 = """
    # drop table training_output.selected_model_performance;
    # """
    
    # sql_query_2 = """
    # SELECT * FROM 
    # (select symbol, avg(avg_valid_accuracy_high_confidence) as llm_selector from training_output.selected_model_performance_v2 group by symbol) a
    # inner join
    # (select symbol, avg(avg_valid_accuracy_high_confidence) as manual_selection from training_output.selected_model_performance group by symbol) b
    # on a.symbol = b.symbol
    # """
    
    sql_query_2 = """
    SELECT * FROM training_data.topic_summary
    """
    
    # sql_query_2 = """
    # DELETE FROM training_output.selected_models
    # WHERE symbol IN ('XLE', 'SPY')
    # OR (symbol = 'QQQ' AND combo_id = 105);
    # """
    
    # sql_query_2 = """
    # DELETE FROM training_output.selected_models
    # WHERE (symbol = 'XLE' AND combo_id = 41) OR (symbol = 'XLE' AND combo_id = 2);
    # """

    # sql_query_2 = """
    # DELETE FROM training_output.selected_model_performance
    # WHERE symbol IN ('XLE', 'TLT');
    # """

    # sql_query_2 = """
    # select * from training_output.selected_models
    # """
    
    rows = dsql_execute_sql(
    host=os.getenv("AWS_AURORA_DB_HOST"),
    database="postgres",
    sql=sql_query_2,
    user="admin",
    region="us-east-1",
    profile="default",
    )
    print(rows)

    # df = pd.DataFrame(
    #     [
    #         ("XLE", 11),
    #         ("XLE", 59),
    # #         ("SPY", 6),
    # #         ("SPY", 129),
    # #         ("SPY", 228),
    #     ],
    #     columns=["symbol", "combo_id"],
    # )

    # print(df)

    # create_table_and_load_df_to_aurora(
    # df=df,
    # host=os.getenv("AWS_AURORA_DB_HOST"),
    # database="postgres",
    # schema_name="training_output",
    # table_name="selected_models",
    # create_table=False,
    # )
