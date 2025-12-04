# Detectors module for TemplateFuzz framework
# Contains jailbreak detection functions for different models

# Note: These modules contain classes with check_jailbreak methods, not standalone functions
# from .jailbreak_detector_deepseek import check_jailbreak_success as check_jailbreak_deepseek
# from .jailbreak_detector_commercial import check_jailbreak_success as check_jailbreak_commercial
# from .jailbreak_detector_qwen3_8B import check_jailbreak_success as check_jailbreak_qwen

# Model-specific detector mapping
DETECTOR_MAPPING = {
    'deepseek': check_jailbreak_deepseek,
    'commercial': check_jailbreak_commercial,
    'qwen': check_jailbreak_qwen,
}

def get_detector_function(model_name):
    """Get the appropriate jailbreak detector for a given model"""
    return DETECTOR_MAPPING.get(model_name.lower(), check_jailbreak_commercial)
