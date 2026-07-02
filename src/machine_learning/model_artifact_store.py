import os
import tempfile
from xgboost import XGBClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.infrastructure.aws_s3 import S3StorageService



class ModelArtifactStore:
    
    def __init__(self):
        self.s3_storage_service = S3StorageService()
        self.client = self.s3_storage_service.client
        self.config = self.s3_storage_service.config


    def load_xgboost_models(self, prefix: str = "models", max_workers: int = 8) -> dict:
        bucket_name = self.config.bucket_name
        if not bucket_name:
            raise ValueError("AWS_S3_BUCKET_NAME is missing")

        paginator = self.client.get_paginator("list_objects_v2")

        s3_keys = []
        for page in paginator.paginate(Bucket=bucket_name, Prefix=f"{prefix}/"):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                if s3_key.endswith(".json"):
                    s3_keys.append(s3_key)

        def load_one_model(s3_key: str):
            file_name = os.path.basename(s3_key)
            model_name = file_name.removesuffix(".json")

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
                temp_file_path = temp_file.name

            try:
                self.client.download_file(bucket_name, s3_key, temp_file_path)
                model = XGBClassifier()
                model.load_model(temp_file_path)
                
                return model_name, model
            
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        loaded_models = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_one_model, s3_key) for s3_key in s3_keys]

            for future in as_completed(futures):
                model_name, model = future.result()
                loaded_models[model_name] = model

        return loaded_models



    def save_xgboost_models(
        self,
        model,
        symbol: str,
        combo_id: int,
        random_state: int,
        if_save_model: bool = False,
        prediction_range: str = "30m",
        prefix: str = "models",
    ) -> None:

        if if_save_model:
            bucket_name = self.config.bucket_name
            if not bucket_name:
                raise ValueError("AWS_S3_BUCKET_NAME is missing")

            file_name = f"{symbol}_{prediction_range}_{combo_id}_{random_state}_XGBoost_Model.json"
            s3_key = f"{prefix}/{symbol}/{prediction_range}/{combo_id}/{file_name}"

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
                temp_file_path = temp_file.name

            try:
                model.save_model(temp_file_path)
                self.client.upload_file(temp_file_path, bucket_name, s3_key)
                
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)