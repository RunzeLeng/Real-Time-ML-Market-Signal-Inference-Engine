import subprocess
import pandas as pd
import json
import boto3
from decimal import Decimal


def dedupe_posts(df: pd.DataFrame, processed_post_ids: set, id_col: str = "id") -> pd.DataFrame:
    """
    Filter out posts whose IDs already exist in the processed_post_ids set.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing posts.
    processed_post_ids : set
        Set of already processed post IDs.
    id_col : str
        Column name containing the post ID.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only new posts.
    """

    # Keep rows whose post_id is NOT in processed_post_ids
    filtered_df = df[~df[id_col].isin(processed_post_ids)].copy()

    return filtered_df.head(1)



def load_df_to_dynamodb_cli(df: pd.DataFrame, table_name: str):

    for _, row in df.iterrows():

        item = {
            "id": {"S": str(row["id"])},
            "created_at": {"S": str(row["created_at"])},
            "content": {"S": str(row["content"])}
        }

        command = [
            "aws",
            "dynamodb",
            "put-item",
            "--table-name",
            table_name,
            "--item",
            json.dumps(item)
        ]

        subprocess.run(command, check=True)
        print(f"Inserted item with ID: {row['id']} into DynamoDB table: {table_name}")



def load_batch_df_to_dynamodb_cli(df: pd.DataFrame, table_name: str) -> None:
    items = []

    for _, row in df.iterrows():
        item = {
            "PutRequest": {
                "Item": {
                    "id": {"S": str(row["id"])},
                    "symbol": {"S": str(row["symbol"])},
                    "created_at": {"S": str(row["created_at"])},
                    "content": {"S": str(row["content"])},
                    "predicted_signal": {"S": str(row["predicted_signal"])},
                    "market_impact_score": {"N": str(row["market_impact_score"])},
                    "reasonableness_score": {"N": str(row["reasonableness_score"])},
                    "brief_reason": {"S": str(row["brief_reason"])},
                    "combined_score": {"N": str(row["combined_score"])},
                    "latency": {"N": str(row["latency"])},
                }
            }
        }
        items.append(item)

    for i in range(0, len(items), 25):
        batch_items = items[i:i + 25]

        request_items = {
            table_name: batch_items
        }

        command = [
            "aws",
            "dynamodb",
            "batch-write-item",
            "--request-items",
            json.dumps(request_items),
        ]

        subprocess.run(command, check=True)
        print(f"Inserted batch {i // 25 + 1} into DynamoDB table: {table_name}")



def load_batch_df_to_dynamodb(df: pd.DataFrame, table_name: str, region_name: str = "us-east-1") -> None:
    dynamodb = boto3.resource("dynamodb", region_name=region_name)
    table = dynamodb.Table(table_name)

    items = []

    for _, row in df.iterrows():
        item = {
            "id": str(row["id"]),
            "symbol": str(row["symbol"]),
            "created_at": str(row["created_at"]),
            "content": str(row["content"]),
            "predicted_signal": str(row["predicted_signal"]),
            "market_impact_score": Decimal(str(row["market_impact_score"])),
            "reasonableness_score": Decimal(str(row["reasonableness_score"])),
            "brief_reason": str(row["brief_reason"]),
            "combined_score": Decimal(str(row["combined_score"])),
            "latency": Decimal(str(row["latency"])),
        }
        items.append(item)

    with table.batch_writer() as batch:
        for item in items:
            batch.put_item(Item=item)

    print(f"Inserted {len(items)} items into DynamoDB table: {table_name}")



def load_ids_from_dynamodb(table_name: str, id_column: str = "id") -> set[str]:
    """
    Load only the ID column from a DynamoDB table into a Python set.

    Parameters
    ----------
    table_name : str
        DynamoDB table name.
    id_column : str
        Attribute name to read from DynamoDB.

    Returns
    -------
    set[str]
        Unique IDs found in the table.
    """
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)

    ids: set[str] = set()
    scan_kwargs = {
        "ProjectionExpression": "#id_attr",
        "ExpressionAttributeNames": {"#id_attr": id_column},
    }

    response = table.scan(**scan_kwargs)

    while True:
        for item in response.get("Items", []):
            item_id = item.get(id_column)
            if item_id is not None:
                ids.add(str(item_id))

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break

        response = table.scan(ExclusiveStartKey=last_evaluated_key, **scan_kwargs)

    return ids



def build_user_prompt_from_post(df):
    """
    Build a user prompt from a one-row DataFrame with columns:
    ID, created_at, and content.

    The prompt starts with the word 'post'.
    """
    content = str(df.iloc[0]["content"]).strip()
    return f"Post: {content}"



def add_id_to_processed_post_ids(df: pd.DataFrame, processed_post_ids: set[str]) -> set[str]:
    if not df.empty:
        processed_post_ids.add(df.iloc[0]["id"])
    
    return processed_post_ids



def input_df_columns_filter(input_df: pd.DataFrame) -> pd.DataFrame:
    json_columns_to_keep = [
        "id",
        "explanation_text",
        *[
            col
            for col in input_df.columns
            if col not in {"id", "user_prompt", "json_output", "explanation_text"}
        ],
    ]

    df_model = input_df[json_columns_to_keep].copy()
    return df_model



def load_dynamodb_table_by_date_range(
    table_name: str,
    start_date: str,
    end_date: str,
    region_name: str = "us-east-1",
) -> pd.DataFrame:
    dynamodb = boto3.resource("dynamodb", region_name=region_name)
    table = dynamodb.Table(table_name)

    response = table.scan()
    items = response.get("Items", [])

    while "LastEvaluatedKey" in response:
        response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        items.extend(response.get("Items", []))

    df = pd.DataFrame(items)

    if df.empty:
        return df

    filtered_df = df[
        (df["created_at"] >= start_date) &
        (df["created_at"] <= end_date)
    ].copy()

    return filtered_df
