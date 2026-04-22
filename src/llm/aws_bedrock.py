import boto3
from pathlib import Path
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import time
import re
import pandas as pd


def concurrent_job_with_prompt_caching_and_dynamic_workers(
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
                result = query_bedrock_model(
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
            save_results_to_jsonl(results, file_save_path)
        
        throttling_ratio = throttled_count / actual_batch_size if actual_batch_size > 0 else 0
        print(f"Batch {batch_start}-{batch_end} throttling ratio: {throttling_ratio:.2f}")

        if throttling_ratio > throttle_ratio_to_reduce:
            current_workers = max(min_workers, current_workers - 1)
        elif throttled_count == 0:
            current_workers = min(max_workers, current_workers + 1)

    return results



def concurrent_job_with_prompt_caching(
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
        result = query_bedrock_model(
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



def query_bedrock_model(
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
    
    client = boto3.client("bedrock-runtime", 
                          region_name=region_name,
                          config=Config(
                            read_timeout=300,
                            connect_timeout=60,
                            retries={"max_attempts": 3, "mode": "standard"},
                          ),
                    )
    
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



def save_results_to_jsonl(results: list[dict], file_path: str = "bedrock_results_single_test.jsonl") -> None:
    Path(file_path).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results),
        encoding="utf-8",
    )



def build_validator_user_prompt(post_df: pd.DataFrame, signal_prediction: pd.DataFrame) -> str:
    post_content = str(post_df.iloc[0]["content"]).strip()

    filtered_predictions = signal_prediction[
        signal_prediction["final_signal"].notna()
    ].copy()

    predicted_signals = filtered_predictions[["symbol", "final_signal"]].rename(
        columns={"final_signal": "predicted_signal"}
    ).to_dict(orient="records")

    predicted_signals_json = json.dumps(predicted_signals, ensure_ascii=False, indent=2)

    user_prompt = f"""
    Evaluate the following post and predicted ETF signals.

    Post content:
    {post_content}

    Predicted ETF signals:
    {predicted_signals_json}

    Please return valid JSON only.
    """
    
    return user_prompt



if __name__ == "__main__":
    
    system_prompt = Path("src/prompt/system_prompt_v2.txt").read_text(encoding="utf-8")

    user_prompt = """
    Post: Iran had plans of taking over the entire Middle East, and completely obliterating Israel. JUST LIKE IRAN ITSELF, THOSE PLANS ARE NOW DEAD! President DONALD J. TRUMP
    """

    result = query_bedrock_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_id="us.anthropic.claude-opus-4-6-v1",
        region_name="us-east-1",
        temperature=0.4,
        max_tokens=2000,
        top_p=0.95,
        top_k=250,
    )

    print("SYSTEM PROMPT:")
    print(system_prompt)
    print()

    print("USER PROMPT:")
    print(user_prompt)
    print()

    print("MODEL OUTPUT:")
    print(result)
    Path("output_bedrock_result.txt").write_text(result, encoding="utf-8")



