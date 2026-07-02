import os
import logging
from src.common.config import setup_logging
from src.live_inference_pipeline.live_inference_pipeline import LiveInferencePipeline

logger = logging.getLogger(__name__)



if __name__ == "__main__":
    
    setup_logging()

    training_version = os.getenv("TRAINING_VERSION", "models")
    logger.info(f"Starting live inference pipeline with training_version = {training_version}")

    pipeline = LiveInferencePipeline()
    pipeline.run(training_version=training_version)