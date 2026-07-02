import random
import time
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config
from botocore.exceptions import ClientError
from src.common.config import LargeLanguageModelConfig, config
from src.processing.json_processing import JSONProcessingService



class LLMService:

    def __init__(
        self,
        llm_config: LargeLanguageModelConfig | None = None,
        json_processing_service: JSONProcessingService | None = None,
    ) -> None:
        
        self.config = llm_config or config.large_language_model
        self.json_processing_service = json_processing_service or JSONProcessingService()
        self.client = self._build_client(self.config.region)
        self.clients_by_region = {self.config.region: self.client}



    def _build_client(self, region_name: str):
        
        return boto3.client(
            "bedrock-runtime",
            region_name=region_name,
            config=Config(
                read_timeout=self.config.read_timeout,
                connect_timeout=self.config.connect_timeout,
                retries={"max_attempts": self.config.max_retry_attempts, "mode": "standard"},
            ),
        )



    def _get_client(self, region_name: str):
        if region_name not in self.clients_by_region:
            self.clients_by_region[region_name] = self._build_client(region_name)

        return self.clients_by_region[region_name]



    def concurrent_job_with_prompt_caching_and_dynamic_workers(
        self,
        ids: list[str],
        user_prompts: list[str],
        system_prompt: str,
        model_id: str,
        region_name: str = "us-east-1",
        temperature: float = 0.4,
        max_tokens: int = 2000,
        top_p: float = 0.95,
        top_k: int = 250,
        stop_sequences: list[str] | None = None,
        initial_workers: int = 2,
        system_prompt_caching: bool = True,
        batch_size: int = 25,
        max_attempts: int = 6,
        min_workers: int = 1,
        max_workers: int = 6,
        throttle_ratio_to_reduce: float = 0.2,
        if_save_file: bool = True,
        file_save_path: str = "bedrock_results_batch_in_progress.jsonl",
    ) -> list[dict]:
        
        results = [None] * len(user_prompts)
        current_workers = initial_workers

        def _run_one(i: int, id: str, prompt: str):
            throttled = False
            
            for attempt in range(max_attempts):
                try:
                    result = self.query_model(
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        model_id=model_id,
                        region_name=region_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        stop_sequences=stop_sequences,
                        system_prompt_caching=system_prompt_caching,
                    )
                    return i, id, prompt, result, throttled

                except ClientError as e:
                    error_code = e.response["Error"]["Code"]

                    if error_code != "ThrottlingException":
                        raise

                    throttled = True

                    if attempt == max_attempts - 1:
                        raise

                    sleep_seconds = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(sleep_seconds)

            raise RuntimeError("Unexpected retry flow in concurrent_running_with_prompt_caching")


        for batch_start in range(0, len(user_prompts), batch_size):
            
            batch_end = min(batch_start + batch_size, len(user_prompts))
            actual_batch_size = batch_end - batch_start
            throttled_count = 0
            print(f"Processing batch {batch_start} to {batch_end} with {current_workers} workers")

            with ThreadPoolExecutor(max_workers=current_workers) as executor:
                futures = [
                    executor.submit(_run_one, i, ids[i], user_prompts[i])
                    for i in range(batch_start, batch_end)
                ]

                for future in as_completed(futures):
                    i, id, prompt, result, throttled = future.result()

                    if throttled:
                        throttled_count += 1

                    results[i] = {
                        "index": i,
                        "id": id,
                        "user_prompt": prompt,
                        "model_output": result,
                    }

            if if_save_file:
                self.json_processing_service.save_result_to_jsonl(results, file_save_path)

            throttling_ratio = throttled_count / actual_batch_size if actual_batch_size > 0 else 0
            print(f"Batch {batch_start}-{batch_end} throttling ratio: {throttling_ratio:.2f}")

            if throttling_ratio > throttle_ratio_to_reduce:
                current_workers = max(min_workers, current_workers - 1)
            elif throttled_count == 0:
                current_workers = min(max_workers, current_workers + 1)

        return results



    def concurrent_job_with_prompt_caching(
        self,
        ids: list[str],
        user_prompts: list[str],
        system_prompt: str,
        model_id: str,
        region_name: str = "us-east-1",
        temperature: float = 0.4,
        max_tokens: int = 2000,
        top_p: float = 0.95,
        top_k: int = 250,
        stop_sequences: list[str] | None = None,
        max_workers: int = 8,
        system_prompt_caching: bool = True,
    ) -> list[dict]:

        results = [None] * len(user_prompts)

        def _run_one(i: int, id: str, prompt: str):
            result = self.query_model(
                system_prompt=system_prompt,
                user_prompt=prompt,
                model_id=model_id,
                region_name=region_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences,
                system_prompt_caching=system_prompt_caching,
            )
            return i, id, prompt, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_run_one, i, ids[i], user_prompts[i])
                for i in range(len(user_prompts))
            ]

            for future in as_completed(futures):
                i, id, prompt, result = future.result()
                results[i] = {
                    "index": i,
                    "id": id,
                    "user_prompt": prompt,
                    "model_output": result,
                }

        return results



    def query_model(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        region_name: str = "us-east-1",
        temperature: float = 0.4,
        max_tokens: int = 2000,
        top_p: float = 0.95,
        top_k: int = 250,
        stop_sequences: list[str] | None = None,
        system_prompt_caching: bool = False,
        include_document: bool = False,
        document_format: str = "csv",
        document_name: str = "attached_file",
        document_bytes: bytes | None = None,
    ) -> str:

        client = self._get_client(region_name)

        if system_prompt:
            if system_prompt_caching:
                system = [
                    {"text": system_prompt},
                    {"cachePoint": {"type": "default"}}
                ]
            else:
                system = [
                    {"text": system_prompt}
                ]
        else:
            system = []

        if include_document:
            user_content = [{"text": user_prompt}]
            user_content.append(
                {
                    "document": {
                        "format": document_format,
                        "name": document_name,
                        "source": {"bytes": document_bytes},
                    }
                }
            )
        else:
            user_content = [{"text": user_prompt}]

        response = client.converse(
            modelId=model_id,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
                # "topP": top_p,
                "stopSequences": stop_sequences or [],
            },
            additionalModelRequestFields={
                "top_k": top_k
            },
        )
        print(response.get("usage"))

        content_blocks = response["output"]["message"]["content"]
        return "".join(block.get("text", "") for block in content_blocks if "text" in block)
