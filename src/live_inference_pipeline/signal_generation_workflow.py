from pathlib import Path
import pandas as pd
from src.prompt.standard_metrics import STANDARD_METRICS
from src.common.exceptions import RestartProcess
from src.infrastructure.aws_dynamodb import DynamoDBService
from src.infrastructure.aws_sns import SnsNotificationService
from src.infrastructure.crawler import CustomCrawlerService
from src.large_language_model.large_language_model import LLMService
from src.large_language_model.prompt_building import LLMPromptBuilder
from src.machine_learning.build_training_data import TrainingDataBuilder
from src.machine_learning.model_registry import ModelRegistry
from src.machine_learning.model_signal_service import ModelSignalService
from src.performance_review.performance_review import PerformanceReviewService
from src.processing.json_processing import JSONProcessingService
from src.processing.post_processing import PostProcessingService



class SignalGenerationWorkflow:

    def __init__(self):
        
        self.crawler_service = CustomCrawlerService()
        self.dynamodb_service = DynamoDBService()
        
        self.post_processing_service = PostProcessingService()
        self.json_processing_service = JSONProcessingService()
        
        self.llm_service = LLMService()
        self.prompt_builder = LLMPromptBuilder()
        
        self.training_data_builder = TrainingDataBuilder()
        self.model_registry = ModelRegistry()
        self.model_signal_service = ModelSignalService()
        
        self.performance_review_service = PerformanceReviewService()
        self.sns_service = SnsNotificationService()



    def crawl_posts_and_preprocess(self, processed_post_ids: set[str]) -> tuple[pd.DataFrame, set[str]]:
        
        df = self.crawler_service.crawl(num_posts=20)
        
        post_df = self.post_processing_service.dedupe_posts(df, processed_post_ids, id_col="id")
        
        if post_df.empty:
            raise RestartProcess("No new posts to process. Restarting the process.")
        else:
            processed_post_ids = self.post_processing_service.add_id_to_processed_post_ids(post_df, processed_post_ids)
        
            post_date = pd.to_datetime(post_df.iloc[0]["created_at"])
            post_start_date = (post_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            post_end_date = (post_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            post_df = self.post_processing_service.filter_posts_by_date_and_content_length(
                df=post_df,
                start_date=post_start_date,
                end_date=post_end_date,
                min_content_length=50,
                date_column="created_at",
                content_column="content",
            )
            
            if post_df.empty:
                raise RestartProcess("No posts after filtering by date and content length. Restarting the process.")
            else:
                post_df = self.post_processing_service.duplicate_posts_to_minute_boundaries(
                    df=post_df,
                    datetime_column="created_at",
                    post_duplicate=False,
                )

                post_df = self.post_processing_service.add_post_prefix_to_content(post_df)
                
                return post_df, processed_post_ids



    def generate_llm_custom_embedding_vector(self, post_df: pd.DataFrame) -> pd.DataFrame:
        
        ids, contents = self.post_processing_service.extract_ids_and_contents(post_df)

        system_prompt = Path("src/prompt/system_prompt_v4.txt").read_text(encoding="utf-8")

        results = self.llm_service.concurrent_job_with_prompt_caching_and_dynamic_workers(
            ids=ids,
            user_prompts=contents,
            system_prompt=system_prompt,
            model_id="us.anthropic.claude-opus-4-6-v1",
            region_name="us-east-1",
            temperature=0.4,
            max_tokens=2000,
            top_p=0.95,
            top_k=250,
            initial_workers=2,
            system_prompt_caching=True,
            batch_size=25,
            max_attempts=6,
            min_workers=1,
            max_workers=3,
            throttle_ratio_to_reduce=0.2,
            if_save_file=False,
            file_save_path="batch_in_progress.jsonl",
        )
        
        output_df = self.json_processing_service.load_single_output_to_df(results)
        
        output_df = self.json_processing_service.expand_metric_json_to_columns(output_df, STANDARD_METRICS)
        
        return output_df



    def ml_model_inference(self, deployed_models: dict, output_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
        
        df_model = self.json_processing_service.columns_filter(output_df)
        
        df_model = self.training_data_builder.scale_input_metric_columns(df_model, STANDARD_METRICS)
        
        X = self.training_data_builder.keep_only_x_and_y_columns(df_model)
        
        selected_model_df = self.model_registry.load_selected_model_performance()
        
        prediction_result = self.model_signal_service.predict_symbol_combo_signals(
            X=X,
            models=deployed_models,
            selected_model_df=selected_model_df,
            max_workers=8,
        )
        
        signal_prediction = self.model_signal_service.symbol_voting_system(prediction_result, post_df)
        
        return signal_prediction



    def llm_validation_and_signal_scoring(self, post_df: pd.DataFrame, signal_prediction: pd.DataFrame) -> pd.DataFrame:
        
        validator_user_prompt = self.prompt_builder.build_validator_user_prompt(post_df, signal_prediction)
        validator_system_prompt = Path("src/prompt/validator_system_prompt.txt").read_text(encoding="utf-8")
        
        validation_result = self.llm_service.query_model(
            system_prompt=validator_system_prompt,
            user_prompt=validator_user_prompt,
            model_id="us.anthropic.claude-opus-4-6-v1",
            region_name="us-east-1",
            temperature=0.2,
            max_tokens=1000,
            top_p=0.95,
            top_k=250,
            system_prompt_caching=False,
        )
        
        validation_df = self.json_processing_service.validator_output_to_df(validation_result)
        
        print(signal_prediction[["id", "symbol", "final_signal"]].to_dict(orient="records"))
        print(validation_df[["symbol", "predicted_signal"]].to_dict(orient="records"))
        print(post_df[["id", "created_at_seconds", "content"]].to_dict(orient="records"))
        
        merged_df = self.model_signal_service.merge_post_signal_and_validation_dfs(signal_prediction, post_df, validation_df)
        
        merged_df = self.model_signal_service.calculate_processing_latency(merged_df)
        
        self.dynamodb_service.save_processed_records(merged_df)
        
        merged_df = self.model_signal_service.score_decision_layer(
            df=merged_df,
            symbol_threshold=0.7,
            combined_threshold=0.4,
        )
        
        if merged_df.empty:
            raise RestartProcess("No valid signals after scoring. Restarting the process.")
        else:
            self.dynamodb_service.save_published_records(merged_df)
            
            return merged_df



    def publish_signals(self, merged_df: pd.DataFrame) -> None:
        
        merged_df = self.performance_review_service.merge_overall_model_accuracy(merged_df)
        
        response = self.sns_service.publish_etf_signals(merged_df)
        
        print(f"SNS publish response: {response}")