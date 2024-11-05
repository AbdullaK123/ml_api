# ml_api/model_manager.py
import torch
import gc
from transformers import pipeline
from paddleocr import PaddleOCR
import logging
import os
from typing import Optional
from contextlib import contextmanager
import threading
import time

logger = logging.getLogger(__name__)

class MemoryMonitor(threading.Thread):
    def __init__(self, threshold_mb=3500):
        super().__init__()
        self.threshold_mb = threshold_mb
        self._stop_event = threading.Event()
        
    def stop(self):
        self._stop_event.set()
        
    def run(self):
        while not self._stop_event.is_set():
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**2
                if memory_used > self.threshold_mb:
                    logger.warning(f"High memory usage detected: {memory_used:.2f}MB")
                    ModelManager().cleanup()
            time.sleep(30)  # Check every 30 seconds

class ModelManager:
    _instance: Optional['ModelManager'] = None
    _initialized: bool = False
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._background_remover = None
            self._ocr_model = None
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._last_used = {}
            self._model_timeout = 300  # 5 minutes
            self._setup_optimizations()
            self._memory_monitor = MemoryMonitor()
            self._memory_monitor.start()
            ModelManager._initialized = True
            logger.info(f"ModelManager initialized using device: {self._device}")

    def _setup_optimizations(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Use less GPU memory
            torch.cuda.memory.set_per_process_memory_fraction(0.7)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            # Enable TensorFloat-32 for better performance/memory trade-off
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        torch.set_num_threads(2)  # Reduce CPU threads
        os.makedirs('/app/model_cache', exist_ok=True)

    @contextmanager
    def model_context(self, model_type: str):
        """Context manager for safe model usage with automatic cleanup"""
        try:
            if model_type == 'background_remover':
                model = self.get_background_remover()
            else:
                model = self.get_ocr_model()
            self._last_used[model_type] = time.time()
            yield model
        finally:
            self._check_idle_cleanup()

    def _check_idle_cleanup(self):
        """Clean up models that haven't been used recently"""
        current_time = time.time()
        with self._lock:
            for model_type, last_used in list(self._last_used.items()):
                if current_time - last_used > self._model_timeout:
                    if model_type == 'background_remover':
                        self._background_remover = None
                    else:
                        self._ocr_model = None
                    del self._last_used[model_type]
                    self._cleanup_memory()

    def _load_background_remover(self):
        with self._lock:
            try:
                self._cleanup_memory()
                self._background_remover = pipeline(
                    "image-segmentation",
                    model="briaai/RMBG-1.4",
                    trust_remote_code=True,
                    device=self._device,
                    model_kwargs={
                        "torch_dtype": torch.float16 if self._device == "cuda" else torch.float32
                    }
                )
            except Exception as e:
                logger.error(f"Error loading background remover: {e}")
                raise

    def _load_ocr_model(self):
        with self._lock:
            try:
                self._cleanup_memory()
                self._ocr_model = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=self._device == "cuda",
                    enable_mkldnn=True,
                    cpu_threads=2,
                    enable_mem_optim=True,
                    max_batch_size=1,
                    use_tensorrt=False,
                    limit_type='max',
                    limit_side_len=2048,  # Reduced from 4096
                    det_db_score_mode='fast',
                    det_limit_side_len=960,  # Reduced from 1280
                    det_db_box_thresh=0.6,
                    rec_batch_num=1
                )
            except Exception as e:
                logger.error(f"Error loading OCR model: {e}")
                raise

    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_background_remover(self):
        if self._background_remover is None:
            self._load_background_remover()
        return self._background_remover

    def get_ocr_model(self):
        if self._ocr_model is None:
            self._load_ocr_model()
        return self._ocr_model

    def cleanup(self):
        with self._lock:
            if self._background_remover is not None:
                del self._background_remover
                self._background_remover = None
            
            if self._ocr_model is not None:
                del self._ocr_model
                self._ocr_model = None
            
            self._last_used.clear()
            self._cleanup_memory()
            logger.info("Models cleaned up")

    def __del__(self):
        self._memory_monitor.stop()
        self.cleanup()

# Global instance
model_manager = ModelManager()