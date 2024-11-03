# ml_api/model_manager.py
import torch
import gc
from transformers import pipeline
from paddleocr import PaddleOCR
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

class ModelManager:
    _instance: Optional['ModelManager'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._background_remover = None
            self._ocr_model = None
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._setup_optimizations()
            ModelManager._initialized = True
            logger.info(f"ModelManager initialized using device: {self._device}")

    def _setup_optimizations(self):
        """Setup memory and performance optimizations"""
        # Setup CUDA optimizations if available
        if torch.cuda.is_available():
            # Set GPU memory usage limits
            torch.cuda.empty_cache()
            torch.cuda.memory.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory
            
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        
        # Set number of threads for CPU operations
        torch.set_num_threads(4)  # Adjust based on your CPU
        
        # Create cache directory
        os.makedirs('/app/model_cache', exist_ok=True)

    def _load_background_remover(self):
        """Lazy load background remover with optimizations"""
        try:
            self._cleanup_memory()
            self._background_remover = pipeline(
                "image-segmentation",
                model="briaai/RMBG-1.4",
                trust_remote_code=True,
                device=self._device,
                model_kwargs={
                    "torch_dtype": torch.float16 if self._device == "cuda" else torch.float32,
                }
            )
            logger.info("Background remover loaded successfully")
        except Exception as e:
            logger.error(f"Error loading background remover: {e}")
            raise

    def _load_ocr_model(self):
        """Lazy load OCR model with optimizations"""
        try:
            self._cleanup_memory()
            self._ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=self._device == "cuda",
                enable_mkldnn=True,
                cpu_threads=4,
                enable_mem_optim=True,
                max_batch_size=1,
                use_tensorrt=False,
                limit_type='max',
                limit_side_len=4096,
                det_db_score_mode='fast',
                det_limit_side_len=1280,  
                det_db_box_thresh=0.6,    
                rec_batch_num=1           
            )
            logger.info("OCR model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading OCR model: {e}")
            raise

    def _cleanup_memory(self):
        """Clean up memory before loading new models"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_background_remover(self):
        """Get background remover with lazy loading"""
        if self._background_remover is None:
            self._load_background_remover()
        return self._background_remover

    def get_ocr_model(self):
        """Get OCR model with lazy loading"""
        if self._ocr_model is None:
            self._load_ocr_model()
        return self._ocr_model

    def cleanup(self):
        """Explicit cleanup method"""
        if self._background_remover is not None:
            del self._background_remover
            self._background_remover = None
        
        if self._ocr_model is not None:
            del self._ocr_model
            self._ocr_model = None
        
        self._cleanup_memory()
        logger.info("Models cleaned up")

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()

# Global instance
model_manager = ModelManager()