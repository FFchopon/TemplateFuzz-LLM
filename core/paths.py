"""
项目默认路径配置（相对于项目根目录）。

所有路径均通过 resolve_path() 解析为绝对路径，避免因工作目录不同导致找不到文件。
可通过环境变量覆盖默认路径：
  - ADVBENCH_DATASET_PATH
  - MMLU_DATASET_PATH
  - LLM_DIR
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 相对路径常量（与 README 保持一致）
ADVBENCH_REL_PATH = "dataset/AdvBench/data/train-00000-of-00001.parquet"
MMLU_REL_PATH = "dataset/mmlu/all/test-00000-of-00001.parquet"
LLM_DIR_REL = "LLM"

SUPPORTED_MODELS = [
    "Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3-70B-Instruct",
    "Llama-2-7b-chat-hf",
    "Qwen2.5-14B-Instruct",
    "Qwen3-8B",
    "deepseek-llm-7b-chat",
    "gpt-oss-20b",
    "gemma-3-4b-it",
    "gemma-3-27b-it",
]


def resolve_path(relative_path: str) -> str:
    """将相对项目根目录的路径解析为绝对路径。"""
    return os.path.normpath(os.path.join(PROJECT_ROOT, relative_path))


def _resolve_with_env(env_var: str, default_rel_path: str) -> str:
    """优先使用环境变量，否则回退到项目内默认相对路径。"""
    override = os.environ.get(env_var)
    if override:
        return os.path.normpath(os.path.abspath(override))
    return resolve_path(default_rel_path)


def get_advbench_path() -> str:
    return _resolve_with_env("ADVBENCH_DATASET_PATH", ADVBENCH_REL_PATH)


def get_mmlu_path() -> str:
    return _resolve_with_env("MMLU_DATASET_PATH", MMLU_REL_PATH)


def get_llm_dir() -> str:
    override = os.environ.get("LLM_DIR")
    if override:
        return os.path.normpath(os.path.abspath(override))
    return resolve_path(LLM_DIR_REL)


def get_model_path(model_name: str) -> str:
    return os.path.join(get_llm_dir(), model_name)


def build_model_paths() -> dict:
    return {name: get_model_path(name) for name in SUPPORTED_MODELS}


# 模块级绝对路径，供各脚本直接引用
ADVBENCH_PARQUET_FILE = get_advbench_path()
MMLU_PARQUET_FILE = get_mmlu_path()
MODEL_PATHS = build_model_paths()
