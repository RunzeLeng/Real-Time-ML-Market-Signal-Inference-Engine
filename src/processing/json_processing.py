import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime



class JSONProcessingService:

    def columns_filter(self, input_df: pd.DataFrame) -> pd.DataFrame:
        json_columns_to_keep = [
            "id",
            "explanation_text",
            *[
                col
                for col in input_df.columns
                if col not in {
                    "id",
                    "user_prompt",
                    "json_output",
                    "explanation_text",
                }
            ],
        ]

        return input_df[json_columns_to_keep].copy()



    def save_result_to_jsonl(
        self,
        results: list[dict],
        file_path: str = "bedrock_results_single_test.jsonl",
    ) -> None:
        
        Path(file_path).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in results),
            encoding="utf-8",
        )



    def extract_last_json_object(self, text: str) -> dict | None:
        candidates = []
        start_positions = [i for i, ch in enumerate(text) if ch == "{"]

        for start in start_positions:
            brace_count = 0
            for end in range(start, len(text)):
                if text[end] == "{":
                    brace_count += 1
                elif text[end] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[start:end + 1]
                        try:
                            candidates.append(json.loads(candidate))
                        except Exception:
                            pass
                        break

        return candidates[-1] if candidates else None



    def extract_last_json_object_with_keyword(self, text: str, json_keyword: str) -> dict | None:
        candidates = []
        start_positions = [i for i, ch in enumerate(text) if ch == "{"]

        for start in start_positions:
            brace_count = 0
            for end in range(start, len(text)):
                if text[end] == "{":
                    brace_count += 1
                elif text[end] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[start:end + 1]
                        try:
                            candidates.append(json.loads(candidate))
                        except Exception:
                            pass
                        break

        candidates_with_keyword = [
            candidate for candidate in candidates
            if json_keyword in candidate
        ]

        return candidates_with_keyword[-1] if candidates_with_keyword else None



    def load_batch_output_to_df(self, folder_path: str = ".") -> pd.DataFrame:
        rows = []

        for file_path in sorted(Path(folder_path).glob("batch_finish*.jsonl")):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        model_output = str(record.get("model_output", "")).strip()

                        json_match = re.search(r"\{.*\}", model_output, re.DOTALL)

                        if json_match:
                            explanation_text = model_output[:json_match.start()].strip()
                            json_output = json_match.group(0).strip()
                        else:
                            explanation_text = model_output
                            json_output = None

                        parsed_json = self.extract_last_json_object(json_output)

                        rows.append({
                            "id": record.get("id"),
                            "user_prompt": record.get("user_prompt"),
                            "explanation_text": explanation_text,
                            "json_output": json.dumps(parsed_json, ensure_ascii=False) if parsed_json is not None else None,
                        })

        return pd.DataFrame(rows, columns=["id", "user_prompt", "explanation_text", "json_output"])



    def load_single_output_to_df(self, result: list[dict]) -> pd.DataFrame:
        if not result:
            return pd.DataFrame(columns=["id", "user_prompt", "explanation_text", "json_output"])

        rows = []
        record = result[0]
        
        model_output = str(record.get("model_output", "")).strip()
        json_match = re.search(r"\{.*\}", model_output, re.DOTALL)

        if json_match:
            explanation_text = model_output[:json_match.start()].strip()
            json_output = json_match.group(0).strip()
        else:
            explanation_text = model_output
            json_output = None

        parsed_json = self.extract_last_json_object(json_output)

        rows.append({
            "id": record.get("id"),
            "user_prompt": record.get("user_prompt"),
            "explanation_text": explanation_text,
            "json_output": json.dumps(parsed_json, ensure_ascii=False) if parsed_json is not None else None,
        })

        return pd.DataFrame(rows, columns=["id", "user_prompt", "explanation_text", "json_output"])



    def validator_output_to_df(self, output_text: str) -> pd.DataFrame:
        cleaned_text = output_text.strip().replace("```json", "").replace("```", "").strip()
        parsed_output = json.loads(cleaned_text)
        
        if parsed_output is None:
            raise ValueError("No valid JSON object found in validator output")

        market_impact_score = parsed_output.get("market_impact_score")
        signal_evaluations = parsed_output.get("signal_evaluations", [])

        rows = []
        for item in signal_evaluations:
            rows.append({
                "symbol": item.get("symbol"),
                "predicted_signal": item.get("predicted_signal"),
                "reasonableness_score": item.get("reasonableness_score"),
                "brief_reason": item.get("brief_reason"),
                "market_impact_score": market_impact_score,
            })

        return pd.DataFrame(
            rows,
            columns=[
                "symbol",
                "predicted_signal",
                "reasonableness_score",
                "brief_reason",
                "market_impact_score",
            ],
        )



    def model_selection_output_to_df(self, output_text: str) -> pd.DataFrame:
        cleaned_text = output_text.strip().replace("```json", "").replace("```", "").strip()
        parsed_output = json.loads(cleaned_text)

        if parsed_output is None:
            raise ValueError("No valid JSON object found in model selection output")

        rows = []
        for _, item in parsed_output.items():
            rows.append({
                "symbol": item.get("symbol"),
                "combo_id": item.get("combo_id"),
                "reason": item.get("reason"),
            })

        return pd.DataFrame(
            rows,
            columns=[
                "symbol",
                "combo_id",
                "reason",
            ],
        )



    def topic_summary_output_to_df(
        self,
        topic: str,
        processing_date: str,
        output_text: str,
    ) -> pd.DataFrame:
        parsed_output = self.extract_last_json_object_with_keyword(output_text, "overall_summary")

        if parsed_output is None:
            raise ValueError("No valid JSON object found in topic summary output")

        row = {
            "topic": topic,
            "processing_date": processing_date,
            "overall_summary": parsed_output.get("overall_summary", ""),
            "seven_day_summary": parsed_output.get("seven_day_summary", ""),
            "three_day_summary": parsed_output.get("three_day_summary", ""),
        }

        return pd.DataFrame(
            [row],
            columns=[
                "topic",
                "processing_date",
                "overall_summary",
                "seven_day_summary",
                "three_day_summary",
            ],
        )



    def news_topic_matching_output_to_df(
        self,
        output_text: str,
        uuid: str,
        title: str,
        published_at: datetime,
        source: str,
    ) -> pd.DataFrame:
        parsed_output = self.extract_last_json_object_with_keyword(output_text, "matched_topics")

        if parsed_output is None:
            raise ValueError("No valid JSON object found in topic matching output")

        matched_topics = parsed_output.get("matched_topics", [])
        rows = []
        
        for item in matched_topics:
            rows.append({
                "uuid": uuid,
                "title": title,
                "published_at": published_at,
                "source": source,
                "topic": item.get("topic_name"),
                "confidence_score": item.get("confidence_score"),
                "reason": item.get("reason"),
            })

        return pd.DataFrame(
            rows,
            columns=[
                "uuid",
                "title",
                "published_at",
                "source",
                "topic",
                "confidence_score",
                "reason",
            ],
        )



    def post_topic_matching_output_to_df(
        self,
        output_text: str,
        id: str,
        post: str,
    ) -> pd.DataFrame:
        parsed_output = self.extract_last_json_object_with_keyword(output_text, "matched_topics")

        if parsed_output is None:
            raise ValueError("No valid JSON object found in topic matching output")

        matched_topics = parsed_output.get("matched_topics", [])
        rows = []

        for item in matched_topics:
            rows.append({
                "id": id,
                "post": post,
                "topic": item.get("topic_name"),
                "confidence_score": item.get("confidence_score"),
                "reason": item.get("reason"),
            })

        return pd.DataFrame(
            rows,
            columns=[
                "id",
                "post",
                "topic",
                "confidence_score",
                "reason",
            ],
        )



    def expand_metric_json_to_columns(
        self,
        df: pd.DataFrame,
        standard_metrics: dict,
    ) -> pd.DataFrame:
        expanded_rows = []

        for _, row in df.iterrows():
            metric_row = standard_metrics.copy()

            json_output = row.get("json_output")
            if json_output:
                metric_values = json.loads(json_output)

                filtered_metric_values = {
                key: value
                for key, value in metric_values.items()
                if key in standard_metrics
                }

                metric_row.update(filtered_metric_values)

            expanded_rows.append(metric_row)

        metrics_df = pd.DataFrame(expanded_rows)

        return pd.concat(
            [
                df[["id", "user_prompt", "explanation_text", "json_output"]].reset_index(drop=True),
                metrics_df.reset_index(drop=True),
            ],
            axis=1,
        )



    def join_etf_with_json_output(
        self,
        etf_df: pd.DataFrame,
        json_df: pd.DataFrame,
    ) -> pd.DataFrame:
        
        json_columns_to_keep = [
            "id",
            "explanation_text",
            *[
                col for col in json_df.columns
                if col not in {"user_prompt", "json_output", "explanation_text"}
                and col not in etf_df.columns
            ],
        ]
        
        etf_df = etf_df.drop(columns=["timestamp", "created_at"]).copy()
        etf_df = etf_df.rename(columns={"created_at_seconds": "created_at"})

        json_subset = json_df[json_columns_to_keep].copy()

        joined_df = etf_df.merge(
            json_subset,
            how="inner",
            on="id",
        )

        return joined_df