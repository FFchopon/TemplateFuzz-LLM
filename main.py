#!/usr/bin/env python3
"""
TemplateFuzz Framework - Main Entry Point
=========================================

This is the main entry point for the TemplateFuzz framework.
It provides convenient access to all major components.

Usage:
    # Standard attack mode
    python main.py --help
    python main.py --model_name Llama-2-7b-chat-hf --bandit_strategy
    
    # Masking mode
    python main.py --mask_mode --model Meta-Llama-3-8B-Instruct --mutation_types M1 M4
    
    # Mutation mode (masking + LLM mutation)
    python main.py --mutation_mode --model Meta-Llama-3-8B-Instruct --mutation_types M1 M5
    
    # Attack mode (masking + LLM mutation + attack target model)
    python main.py --attack_mode --model Meta-Llama-3-8B-Instruct --mutation_types M1 M5
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main entry point; selects run mode based on arguments."""
    # Check for baseline test mode
    if "--baseline_test" in sys.argv:
        # Remove --baseline_test to avoid passing it to baseline_test module
        sys.argv.remove("--baseline_test")
        from core.baseline_test import main as baseline_test_main
        baseline_test_main()
    # Check for attack mode
    elif "--attack_mode" in sys.argv:
        # Remove --attack_mode to avoid passing it to attack_mode module
        sys.argv.remove("--attack_mode")
        from core.attack_mode import main as attack_main
        attack_main()
    # Check for mutation mode
    elif "--mutation_mode" in sys.argv:
        # Remove --mutation_mode to avoid passing it to mutation_mode module
        sys.argv.remove("--mutation_mode")
        from core.mutation_mode import main as mutation_main
        mutation_main()
    # Check for masking mode
    elif "--mask_mode" in sys.argv:
        # Remove --mask_mode to avoid passing it to mask_mode module
        sys.argv.remove("--mask_mode")
        from core.mask_mode import main as mask_main
        mask_main()
    else:
        # Standard attack mode
        from core.baseline import main as baseline_main
        baseline_main()

if __name__ == "__main__":
    main()
