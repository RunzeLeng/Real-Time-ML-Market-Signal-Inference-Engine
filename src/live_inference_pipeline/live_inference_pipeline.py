import time
from dotenv import load_dotenv
from src.common.exceptions import RestartProcess
from src.infrastructure.crawler import CustomCrawlerService
from src.live_inference_pipeline.signal_generation_workflow import SignalGenerationWorkflow
from src.machine_learning.model_artifact_store import ModelArtifactStore
from src.performance_review.performance_review import PerformanceReviewService



class LiveInferencePipeline:

    def __init__(self):
        
        self.crawler_service = CustomCrawlerService()
        self.model_artifact_store = ModelArtifactStore()
        
        self.signal_generation_workflow = SignalGenerationWorkflow()
        self.performance_review_service = PerformanceReviewService()



    def inference_init(self, training_version: str = "models"):
        
        deployed_models = self.model_artifact_store.load_xgboost_models(prefix=training_version)

        df = self.crawler_service.crawl(num_posts=20)
        processed_post_ids = set(df["id"].dropna())
        
        return deployed_models, processed_post_ids



    def run(self, training_version: str = "models") -> None:
        
        load_dotenv()
        print("Starting inference pipeline...", flush=True)
        
        self.performance_review_service.run_review()
        
        deployed_models, processed_post_ids = self.inference_init(training_version=training_version)
        
        while True:
            try:
                # processed_post_ids.discard("116845625781087003")
                
                post_df, processed_post_ids = self.signal_generation_workflow.crawl_posts_and_preprocess(processed_post_ids)
                
                output_df = self.signal_generation_workflow.generate_llm_custom_embedding_vector(post_df)
                
                signal_prediction = self.signal_generation_workflow.ml_model_inference(deployed_models, output_df, post_df)
                
                merged_df = self.signal_generation_workflow.llm_validation_and_signal_scoring(post_df, signal_prediction)   

                self.signal_generation_workflow.publish_signals(merged_df)
                
                print("Process completed successfully.")

            except RestartProcess as e:
                
                print(f"Restarting process: {e}", flush=True)
                time.sleep(10)
                continue