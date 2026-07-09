import logging
import os
from vllm import LLM
from transformers import AutoTokenizer
from .paths import get_model_path

def setup_logging():
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()  # 只保留控制台输出，移除文件日志
        ]
    )

def load_model(model_name):
    """
    加载模型和分词器
    Args:
        model_name: 模型名称
    Returns:
        tuple: (model, tokenizer)
    """
    model_path = get_model_path(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径 {model_path} 不存在")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.3, enforce_eager=True)
        logging.info(f"成功加载模型 {model_name}")
        return model, tokenizer
    except Exception as e:
        logging.error(f"加载模型 {model_name} 失败: {e}")
        raise

