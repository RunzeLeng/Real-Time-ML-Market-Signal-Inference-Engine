import json
import subprocess
from decimal import Decimal
import boto3
import pandas as pd
from src.common.config import config



class DynamoDBService:
    
    def __init__(self):
        self.config = config.dynamodb
        self.dynamodb = boto3.resource(
            "dynamodb",
            region_name=self.config.region,
        )



    def table(self, table_name: str):
        return self.dynamodb.Table(table_name)



    def save_processed_records(self, df: pd.DataFrame) -> None:
        self.load_batch_df_to_dynamodb(
            df=df,
            table_name=self.config.processed_records_table,
        )



    def save_published_records(self, df: pd.DataFrame) -> None:
        self.load_batch_df_to_dynamodb(
            df=df,
            table_name=self.config.published_records_table,
        )



    def load_batch_df_to_dynamodb(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> None:
        table = self.table(table_name)

        items = [
            self._build_signal_item(row)
            for _, row in df.iterrows()
        ]

        with table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)

        print(f"Inserted {len(items)} items into DynamoDB table: {table_name}")



    def load_ids_from_dynamodb(
        self,
        table_name: str,
        id_column: str = "id",
    ) -> set[str]:
        table = self.table(table_name)

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

            response = table.scan(
                ExclusiveStartKey=last_evaluated_key,
                **scan_kwargs,
            )

        return ids



    def load_table_by_date_range(
        self,
        table_name: str,
        start_date: str,
        end_date: str,
        date_column: str = "created_at",
    ) -> pd.DataFrame:
        table = self.table(table_name)

        response = table.scan()
        items = response.get("Items", [])

        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response.get("Items", []))

        df = pd.DataFrame(items)

        if df.empty:
            return df

        return df[
            (df[date_column] >= start_date)
            & (df[date_column] <= end_date)
        ].copy()



    def load_published_records_by_date_range(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return self.load_table_by_date_range(
            table_name=self.config.published_records_table,
            start_date=start_date,
            end_date=end_date,
        )



    @staticmethod
    def _build_signal_item(row: pd.Series) -> dict:
        return {
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




def legacy_load_df_to_dynamodb_cli(df: pd.DataFrame, table_name: str):
    for _, row in df.iterrows():
        item = {
            "id": {"S": str(row["id"])},
            "created_at": {"S": str(row["created_at"])},
            "content": {"S": str(row["content"])},
        }

        command = [
            "aws",
            "dynamodb",
            "put-item",
            "--table-name",
            table_name,
            "--item",
            json.dumps(item),
        ]

        subprocess.run(command, check=True)
        print(f"Inserted item with ID: {row['id']} into DynamoDB table: {table_name}")



def legacy_load_batch_df_to_dynamodb_cli(df: pd.DataFrame, table_name: str) -> None:
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
            table_name: batch_items,
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
