# Evaluate module for TemplateFuzz framework
# Contains evaluation functions for different models and scenarios

from .evaluate_with_templates import evaluate_template
# evaluate_commercial_api module contains various evaluation functions
from .evaluate_qwen import evaluate_template as evaluate_qwen
from .evaluate_llama2 import evaluate_template as evaluate_llama2
from .evaluate_llama3 import evaluate_template as evaluate_llama3
from .evaluate_deepseek import evaluate_template as evaluate_deepseek
from .analyze_attack_results import analyze_attack_results

# Model-specific evaluation mapping
EVALUATION_MAPPING = {
    'qwen': evaluate_qwen,
    'llama2': evaluate_llama2,
    'llama3': evaluate_llama3,
    'deepseek': evaluate_deepseek,
}

def get_evaluation_function(model_name):
    """Get the appropriate evaluation function for a given model"""
    return EVALUATION_MAPPING.get(model_name.lower(), evaluate_template)
