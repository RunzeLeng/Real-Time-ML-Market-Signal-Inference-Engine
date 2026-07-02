import os
import logging
from src.common.config import setup_logging
from src.machine_learning.model_retraining_workflow import ModelRetrainingWorkflow

logger = logging.getLogger(__name__)



if __name__ == "__main__":
    
    setup_logging()

    training_version = os.getenv("TRAINING_VERSION", "models")
    logger.info(f"Starting model retraining pipeline, new training_version = {training_version}")

    workflow = ModelRetrainingWorkflow()

    workflow.run_auto_retraining_pipeline(
        training_version=training_version,
        ECS_ETF_LIST_OVERRIDE=workflow.apply_ecs_etf_override(),
    )

    logger.info("Model retraining pipeline completed")