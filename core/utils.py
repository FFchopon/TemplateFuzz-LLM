import logging
import os
from vllm import LLM
from transformers import AutoTokenizer
from .paths import get_model_path

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()  # Console output only; file logging removed
        ]
    )

def load_model(model_name):
    """
    Load model and tokenizer.
    Args:
        model_name: Model name
    Returns:
        tuple: (model, tokenizer)
    """
    model_path = get_model_path(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.3, enforce_eager=True)
        logging.info(f"Successfully loaded model {model_name}")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {e}")
        raise
