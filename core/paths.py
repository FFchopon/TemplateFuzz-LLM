"""
Default project path configuration (relative to project root).

All paths are resolved to absolute paths via resolve_path() to avoid file-not-found
errors when the working directory differs.
Default paths can be overridden via environment variables:
  - ADVBENCH_DATASET_PATH
  - MMLU_DATASET_PATH
  - LLM_DIR
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Relative path constants (consistent with README)
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
    """Resolve a path relative to project root to an absolute path."""
    return os.path.normpath(os.path.join(PROJECT_ROOT, relative_path))


def _resolve_with_env(env_var: str, default_rel_path: str) -> str:
    """Prefer environment variable; fall back to default relative path within project."""
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


# Module-level absolute paths for direct import by scripts
ADVBENCH_PARQUET_FILE = get_advbench_path()
MMLU_PARQUET_FILE = get_mmlu_path()
MODEL_PATHS = build_model_paths()
