#!/usr/bin/env python3
"""
TemplateFuzz Framework - Main Entry Point
=========================================

This is the main entry point for the TemplateFuzz framework.
It provides convenient access to all major components.

Usage:
    # 标准攻击模式
    python main.py --help
    python main.py --model_name Llama-2-7b-chat-hf --bandit_strategy
    
    # 挖空模式
    python main.py --mask_mode --model Meta-Llama-3-8B-Instruct --mutation_types M1 M4
    
    # 变异模式（挖空+LLM变异）
    python main.py --mutation_mode --model Meta-Llama-3-8B-Instruct --mutation_types M1 M5
    
    # 攻击模式（挖空+LLM变异+攻击目标模型）
    python main.py --attack_mode --model Meta-Llama-3-8B-Instruct --mutation_types M1 M5
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """主入口函数，根据参数选择运行模式"""
    # 检查是否使用原始模板测试模式
    if "--baseline_test" in sys.argv:
        # 移除 --baseline_test 参数，避免传递给 baseline_test 模块
        sys.argv.remove("--baseline_test")
        from core.baseline_test import main as baseline_test_main
        baseline_test_main()
    # 检查是否使用攻击模式
    elif "--attack_mode" in sys.argv:
        # 移除 --attack_mode 参数，避免传递给 attack_mode 模块
        sys.argv.remove("--attack_mode")
        from core.attack_mode import main as attack_main
        attack_main()
    # 检查是否使用变异模式
    elif "--mutation_mode" in sys.argv:
        # 移除 --mutation_mode 参数，避免传递给 mutation_mode 模块
        sys.argv.remove("--mutation_mode")
        from core.mutation_mode import main as mutation_main
        mutation_main()
    # 检查是否使用挖空模式
    elif "--mask_mode" in sys.argv:
        # 移除 --mask_mode 参数，避免传递给 mask_mode 模块
        sys.argv.remove("--mask_mode")
        from core.mask_mode import main as mask_main
        mask_main()
    else:
        # 标准攻击模式
        from core.baseline import main as baseline_main
        baseline_main()

if __name__ == "__main__":
    main()
