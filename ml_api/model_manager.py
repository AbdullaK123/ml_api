# ml_api/model_manager.py
import torch
from transformers import pipeline
from paddleocr import PaddleOCR
import logging
import os

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            logger.info("Initializing models from cache...")
            self._initialize_models()
            ModelManager._initialized = True

    def _initialize_models(self):
        try:
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Load models from cache
            cache_dir = '/app/model_cache'
            os.makedirs(cache_dir, exist_ok=True)

            logger.info("Loading background remover model...")
            self.background_remover = pipeline(
                "image-segmentation",
                model="briaai/RMBG-1.4",
                trust_remote_code=True,
                device="cpu"
            )

            logger.info("Loading OCR model...")
            self.ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=False,
                enable_mkldnn=True,
                cpu_threads=4
            )

            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Error in model initialization: {e}")
            raise

    def get_background_remover(self):
        return self.background_remover

    def get_ocr_model(self):
        return self.ocr_model

# Global model manager instance
model_manager = ModelManager()