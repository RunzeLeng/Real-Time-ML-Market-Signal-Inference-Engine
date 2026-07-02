import os
from pathlib import Path
from src.common.etf_constants import ETF_LIST
from src.prompt.standard_metrics import STANDARD_METRICS
from src.infrastructure.aws_aurora_dsql import AuroraDsqlClient
from src.large_language_model.large_language_model import LLMService
from src.machine_learning.build_training_data import TrainingDataBuilder
from src.machine_learning.model_registry import ModelRegistry
from src.machine_learning.model_training import ModelTrainingService
from src.processing.json_processing import JSONProcessingService



class ModelRetrainingWorkflow:

    def __init__(self):
        
        self.training_data_builder = TrainingDataBuilder()
        self.model_training_service = ModelTrainingService()
        self.model_registry = ModelRegistry()
        self.llm_service = LLMService()
        
        self.json_processing_service = JSONProcessingService()
        self.aurora_client = AuroraDsqlClient()



    def run_hyperparameter_search(
        self,
        training_version: str,
        ECS_ETF_LIST_OVERRIDE: list | None = None,
    ) -> None:
        
        df = self.training_data_builder.load_training_data()
        etf_list = self.resolve_etf_list(ECS_ETF_LIST_OVERRIDE)

        for symbol in etf_list:
            
            filtered_df = df[df["symbol"] == symbol].copy()
            print(f"\nSymbol: {symbol}, count: {len(filtered_df)}")
            
            training_results = self.model_training_service.train_all_hyperparameter_combinations(
                filtered_df,
                symbol,
                STANDARD_METRICS,
                random_state_length=100,
            )
            
            self.aurora_client.create_table_and_load_df_to_aurora(
                df=training_results,
                schema_name="training_output",
                table_name=f"model_performance_{training_version}",
                create_table=(symbol == "QQQ"),
            )



    def select_model_with_llm(
        self,
        training_version: str,
        ECS_ETF_LIST_OVERRIDE: list | None = None,
    ) -> None:
        
        model_performance = self.model_registry.load_all_model_performance(training_version=training_version)
        etf_list = self.resolve_etf_list(ECS_ETF_LIST_OVERRIDE)

        for symbol in etf_list:
            
            filtered_model_performance = model_performance[model_performance["symbol"] == symbol].copy()
            print(f"Selecting symbol: {symbol}")
            
            csv_bytes = filtered_model_performance.to_csv(index=False).encode("utf-8")
            system_prompt = Path("src/prompt/model_selection_system_prompt.txt").read_text(encoding="utf-8")
            user_prompt = Path("src/prompt/model_selection_user_prompt.txt").read_text(encoding="utf-8")
            
            model_selection_result = self.llm_service.query_model(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_id="us.anthropic.claude-opus-4-6-v1",
                region_name="us-east-1",
                temperature=0.2,
                max_tokens=1000,
                top_p=0.95,
                top_k=100,
                system_prompt_caching=False,
                include_document=True,
                document_format="csv",
                document_name="attached_file",
                document_bytes=csv_bytes,
            )
            
            model_selection_df = self.json_processing_service.model_selection_output_to_df(model_selection_result)
            
            self.aurora_client.create_table_and_load_df_to_aurora(
                df=model_selection_df,
                schema_name="training_output",
                table_name=f"selected_models_{training_version}",
                create_table=(symbol == "QQQ"),
            )



    def train_selected_models(
        self,
        training_version: str = "models",
        ECS_ETF_LIST_OVERRIDE: list | None = None,
    ) -> None:
        
        df = self.training_data_builder.load_training_data()
        model_combos = self.model_registry.load_selected_model_combos(training_version=training_version)
        etf_list = self.resolve_etf_list(ECS_ETF_LIST_OVERRIDE)

        for symbol in etf_list:
            
            filtered_df = df[df["symbol"] == symbol].copy()
            filtered_model_combos = model_combos[model_combos["symbol"] == symbol].copy()
            
            print(f"\nSymbol: {symbol}, count: {len(filtered_df)}")
            print(f"\nSymbol: {symbol}, count: {len(filtered_model_combos)}")
            
            training_results = self.model_training_service.train_selected_hyperparameter_combinations(
                filtered_df,
                filtered_model_combos,
                symbol,
                STANDARD_METRICS,
                random_state_length=100,
                prefix=training_version,
            )
            
            self.aurora_client.create_table_and_load_df_to_aurora(
                df=training_results,
                schema_name="training_output",
                table_name=f"selected_model_performance_{training_version}",
                create_table=(symbol == "QQQ"),
            )



    def run_auto_retraining_pipeline(
        self,
        training_version: str,
        ECS_ETF_LIST_OVERRIDE: list | None = None,
    ) -> None:
        
        self.run_hyperparameter_search(
            training_version=training_version, 
            ECS_ETF_LIST_OVERRIDE=ECS_ETF_LIST_OVERRIDE
        )
        
        self.select_model_with_llm(
            training_version=training_version, 
            ECS_ETF_LIST_OVERRIDE=ECS_ETF_LIST_OVERRIDE
        )
        
        self.train_selected_models(
            training_version=training_version, 
            ECS_ETF_LIST_OVERRIDE=ECS_ETF_LIST_OVERRIDE
        )



    def apply_ecs_etf_override(self) -> list[str] | None:
        target_etfs_env = os.getenv("TARGET_ETFS", "").strip()
        
        if not target_etfs_env:
            return None

        ecs_etf_list_override = [
            etf.strip().upper()
            for etf in target_etfs_env.split(",")
            if etf.strip()
        ]

        return ecs_etf_list_override



    def resolve_etf_list(self, ECS_ETF_LIST_OVERRIDE: list | None = None) -> list:
        
        if ECS_ETF_LIST_OVERRIDE:
            etf_list = ECS_ETF_LIST_OVERRIDE
        else:
            etf_list = ETF_LIST

        return etf_list
