#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyze single attack result files.
Processes breached question counts and average attack attempts from JSON files.
"""

import json
import sys
import os
import csv
from datetime import datetime

def analyze_attack_results(json_file_path):
    """
    Analyze attack result file.
    
    Args:
        json_file_path (str): Path to JSON result file
        
    Returns:
        dict: Dictionary containing analysis results
    """
    try:
        # Read JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get basic information
        total_questions = data.get('attack_phase', {}).get('total_questions', 0)
        successful_attacks = data.get('successful_attacks', [])
        failed_attacks = data.get('failed_attacks', [])
        
        # Count breached questions
        successful_count = len(successful_attacks)
        failed_count = len(failed_attacks)
        
        # Extract attack attempts for each successful breach
        attack_attempts = []
        for attack in successful_attacks:
            attempts = attack.get('attempts', 0)
            attack_attempts.append(attempts)
        
        # Compute average attack attempts
        if attack_attempts:
            average_attempts = sum(attack_attempts) / len(attack_attempts)
        else:
            average_attempts = 0
        
        # Build results
        results = {
            'total_questions': total_questions,
            'successful_count': successful_count,
            'failed_count': failed_count,
            'success_rate': successful_count / total_questions if total_questions > 0 else 0,
            'average_attempts': average_attempts,
            'attack_attempts': attack_attempts,
            'min_attempts': min(attack_attempts) if attack_attempts else 0,
            'max_attempts': max(attack_attempts) if attack_attempts else 0
        }
        
        return results
        
    except FileNotFoundError:
        print("Error: File not found {}".format(json_file_path))
        return None
    except json.JSONDecodeError:
        print("Error: Unable to parse JSON file {}".format(json_file_path))
        return None
    except Exception as e:
        print("Error: Exception while processing file - {}".format(str(e)))
        return None

def print_results(results):
    """
    Print analysis results.
    
    Args:
        results (dict): Analysis results dictionary
    """
    if results is None:
        return
    
    print("=" * 60)
    print("Attack Result Analysis Report")
    print("=" * 60)
    print("Total questions: {}".format(results['total_questions']))
    print("Breached questions: {}/{}".format(results['successful_count'], results['total_questions']))
    print("Failed questions: {}".format(results['failed_count']))
    print("Success rate: {:.2%}".format(results['success_rate']))
    print("-" * 40)
    print("Average attack attempts: {:.2f}".format(results['average_attempts']))
    print("Minimum attack attempts: {}".format(results['min_attempts']))
    print("Maximum attack attempts: {}".format(results['max_attempts']))
    print("-" * 40)
    
    # Show attack attempt distribution
    if results['attack_attempts']:
        print("Attack attempt distribution:")
        attempts_count = {}
        for attempts in results['attack_attempts']:
            attempts_count[attempts] = attempts_count.get(attempts, 0) + 1
        
        # Sort by attack attempts
        for attempts in sorted(attempts_count.keys()):
            count = attempts_count[attempts]
            percentage = count / len(results['attack_attempts']) * 100
            print("  {} attempt(s): {} question(s) ({:.1f}%)".format(attempts, count, percentage))
    
    print("=" * 60)

def main():
    """Main entry point."""
    # Default file path
    default_file = "output/Llama-2-7b-chat-hf/single_attack_bandit_strategy/single_attack_results/single_attack_Llama-2-7b-chat-hf_bandit_strategy_20250730_165251.json"
    
    # Check command-line arguments
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        json_file_path = default_file
    
    # Check file exists
    if not os.path.exists(json_file_path):
        print("Error: File does not exist - {}".format(json_file_path))
        print("Usage: python {} [JSON file path]".format(sys.argv[0]))
        print("Default file: {}".format(default_file))
        sys.exit(1)
    
    print("Analyzing file: {}".format(json_file_path))
    print()
    
    # Analyze results
    results = analyze_attack_results(json_file_path)
    
    # Print results
    print_results(results)
    
    # Save comprehensive results to CSV file
    if results:
        base_name = json_file_path.replace('.json', '')
        output_file = "{}_results.csv".format(base_name)
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write overall statistics
                writer.writerow(['Attack Result Analysis Report'])
                writer.writerow(['Analysis time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['Source file', os.path.basename(json_file_path)])
                writer.writerow([])  # Blank row separator
                
                # Overall statistics
                writer.writerow(['Overall Statistics'])
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total questions', results['total_questions']])
                writer.writerow(['Breached questions', results['successful_count']])
                writer.writerow(['Failed questions', results['failed_count']])
                writer.writerow(['Success rate (%)', "{:.2f}".format(results['success_rate']*100)])
                writer.writerow(['Average attack attempts', "{:.2f}".format(results['average_attempts'])])
                writer.writerow(['Minimum attack attempts', results['min_attempts']])
                writer.writerow(['Maximum attack attempts', results['max_attempts']])
                writer.writerow([])  # Blank row separator
                
                # Detailed attack attempt data
                writer.writerow(['Per-question Attack Attempt Details'])
                writer.writerow(['Question index', 'Attack attempts', 'Attempt range'])
                
                for i, attempts in enumerate(results['attack_attempts'], 1):
                    # Define attack attempt ranges
                    if attempts == 1:
                        interval = '1 attempt'
                    elif attempts <= 5:
                        interval = '2-5 attempts'
                    elif attempts <= 10:
                        interval = '6-10 attempts'
                    elif attempts <= 20:
                        interval = '11-20 attempts'
                    elif attempts <= 50:
                        interval = '21-50 attempts'
                    else:
                        interval = '50+ attempts'
                    
                    writer.writerow([i, attempts, interval])
                
                writer.writerow([])  # Blank row separator
                
                # Attack attempt distribution statistics
                writer.writerow(['Attack Attempt Distribution Statistics'])
                writer.writerow(['Attack attempts', 'Question count', 'Percentage (%)'])
                
                attempts_count = {}
                for attempts in results['attack_attempts']:
                    attempts_count[attempts] = attempts_count.get(attempts, 0) + 1
                
                # Sort by attack attempts
                for attempts in sorted(attempts_count.keys()):
                    count = attempts_count[attempts]
                    percentage = count / len(results['attack_attempts']) * 100
                    writer.writerow([attempts, count, "{:.1f}".format(percentage)])
            
            print("\nComprehensive analysis results saved to: {}".format(output_file))
            
        except Exception as e:
            print("Warning: Unable to save CSV results file - {}".format(str(e)))

if __name__ == "__main__":
    main()
