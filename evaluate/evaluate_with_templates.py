#!/usr/bin/env python3
"""
Commercial LLM API evaluation framework - supports template files.
"""

import json
import argparse
import os
from .evaluate_commercial_api import evaluate_template, evaluate_templated_json, batch_process_json_directory, save_results, MODEL_CONFIGS, load_template, is_chat_template
import logging
from datetime import datetime

def load_templates(template_file="templates.json"):
    """Load template configuration from JSON file."""
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['templates']
    except Exception as e:
        logging.error(f"Failed to load template file: {e}")
        return {}

def list_templates(templates):
    """List all available templates."""
    print("Available templates:")
    print("-" * 50)
    for key, template_info in templates.items():
        print(f"ID: {key}")
        print(f"Name: {template_info['name']}")
        print(f"Description: {template_info['description']}")
        print(f"Template: {template_info['template'][:100]}...")
        print("-" * 50)

def batch_evaluate(model_name, template_ids, templates, num_samples=None):
    """Batch evaluate multiple templates."""
    results = []
    
    for template_id in template_ids:
        if template_id not in templates:
            logging.warning(f"Template '{template_id}' does not exist, skipping")
            continue
        
        template_info = templates[template_id]
        template_str = template_info['template']
        template_name = template_info['name']
        
        print(f"\n{'='*60}")
        print(f"Evaluating template: {template_id} ({template_name})")
        print(f"{'='*60}")
        print(f"Template content: {template_str}")
        print()
        
        try:
            jailbreak_rate, experiment_results = evaluate_template(
                model_name, template_str, num_samples, False  # templates.json entries are not file templates
            )
            
            # Prepare result
            result = {
                "model": model_name,
                "template_id": template_id,
                "template_name": template_name,
                "template": template_str,
                "num_samples": num_samples or len(experiment_results),
                "metrics": {
                    "jailbreak_success_rate": jailbreak_rate,
                    "total_responses": len([r for r in experiment_results if r.get("generated_text")])
                },
                "experiment_results": experiment_results,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
            # Save single template result
            save_results(result, model_name, template_id)
            
            # Print summary
            print(f"\n--- {template_id} Result Summary ---")
            print(f"Jailbreak success rate: {jailbreak_rate:.4f}")
            print(f"Total responses: {result['metrics']['total_responses']}")
            
        except Exception as e:
            logging.error(f"Error evaluating template '{template_id}': {e}")
            continue
    
    return results

def batch_evaluate_txt_files(model_name, template_files, num_samples=None):
    """Batch evaluate .txt template files."""
    results = []
    
    for template_file in template_files:
        if not template_file.endswith('.txt'):
            logging.warning(f"Skipping non-.txt file: {template_file}")
            continue
        
        try:
            # Load template file
            template_str, is_file_template = load_template(template_file)
            template_name = os.path.basename(template_file).replace('.txt', '')
            
            print(f"\n{'='*60}")
            print(f"Evaluating template file: {template_file}")
            print(f"{'='*60}")
            if is_chat_template(template_str):
                print("Detected Jinja2 chat template format")
            print(f"Template preview: {template_str[:200]}...")
            print()
            
            jailbreak_rate, experiment_results = evaluate_template(
                model_name, template_str, num_samples, is_file_template
            )
            
            # Prepare result
            result = {
                "model": model_name,
                "template_id": template_name,
                "template_name": template_name,
                "template": template_str,
                "template_source": template_file,
                "num_samples": num_samples or len(experiment_results),
                "metrics": {
                    "jailbreak_success_rate": jailbreak_rate,
                    "total_responses": len([r for r in experiment_results if r.get("generated_text")])
                },
                "experiment_results": experiment_results,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
            # Save single template result
            save_results(result, model_name, template_name)
            
            # Print summary
            print(f"\n--- {template_name} Result Summary ---")
            print(f"Jailbreak success rate: {jailbreak_rate:.4f}")
            print(f"Total responses: {result['metrics']['total_responses']}")
            
        except Exception as e:
            logging.error(f"Error evaluating template file '{template_file}': {e}")
            continue
    
    return results

def batch_evaluate_json_files(model_name, json_files, num_samples=None):
    """Batch evaluate pre-processed JSON files."""
    results = []
    
    for json_file in json_files:
        if not json_file.endswith('.json'):
            logging.warning(f"Skipping non-.json file: {json_file}")
            continue
        
        try:
            template_name = os.path.basename(json_file).replace('.json', '')
            
            print(f"\n{'='*60}")
            print(f"Evaluating JSON file: {json_file}")
            print(f"{'='*60}")
            
            jailbreak_rate, experiment_results = evaluate_templated_json(
                model_name, json_file, num_samples
            )
            
            # Prepare result
            result = {
                "model": model_name,
                "template_id": template_name,
                "template_name": template_name,
                "input_mode": "templated_json",
                "source_json": json_file,
                "num_samples": num_samples or len(experiment_results),
                "metrics": {
                    "jailbreak_success_rate": jailbreak_rate,
                    "total_responses": len([r for r in experiment_results if r.get("generated_text")])
                },
                "experiment_results": experiment_results,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
            # Save single JSON file result
            save_results(result, model_name, template_name)
            
            # Print summary
            print(f"\n--- {template_name} Result Summary ---")
            print(f"Jailbreak success rate: {jailbreak_rate:.4f}")
            print(f"Total responses: {result['metrics']['total_responses']}")
            
        except Exception as e:
            logging.error(f"Error evaluating JSON file '{json_file}': {e}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Commercial LLM API evaluation framework - supports template files")
    parser.add_argument("--model", required=True, 
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model name")
    parser.add_argument("--template-file", default="templates.json",
                       help="Template configuration file path")
    parser.add_argument("--template-ids", nargs="+",
                       help="List of template IDs to evaluate")
    parser.add_argument("--template-txt-files", nargs="+",
                       help="List of .txt template file paths")
    parser.add_argument("--templated-json-files", nargs="+",
                       help="List of JSON file paths containing pre-processed templated_input")
    parser.add_argument("--json-dir", 
                       help="Directory path containing multiple JSON files for batch processing (jailbreak prompts only)")
    parser.add_argument("--list-templates", action="store_true",
                       help="List all available templates")
    parser.add_argument("--num-samples", type=int,
                       help="Number of samples; uses all data if not specified")
    parser.add_argument("--output-summary", 
                       help="Filename to save batch evaluation summary")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load templates
    templates = load_templates(args.template_file)
    if not templates:
        print("Unable to load template file")
        return
    
    # List templates
    if args.list_templates:
        list_templates(templates)
        return
    
    # Check template arguments
    if not args.template_ids and not args.template_txt_files and not args.templated_json_files and not args.json_dir:
        print("Please specify template IDs (--template-ids), template files (--template-txt-files), JSON files (--templated-json-files), or directory (--json-dir), or use --list-templates to view available templates")
        return
    
    print(f"Starting batch evaluation for model: {args.model}")
    
    results = []
    
    # Evaluate templates from JSON template file
    if args.template_ids:
        print(f"Evaluating JSON templates: {args.template_ids}")
        results.extend(batch_evaluate(
            args.model, 
            args.template_ids, 
            templates, 
            args.num_samples
        ))
    
    # Evaluate .txt template files
    if args.template_txt_files:
        print(f"Evaluating .txt template files: {args.template_txt_files}")
        results.extend(batch_evaluate_txt_files(
            args.model, 
            args.template_txt_files, 
            args.num_samples
        ))
    
    # Evaluate pre-processed JSON files
    if args.templated_json_files:
        print(f"Evaluating pre-processed JSON files: {args.templated_json_files}")
        results.extend(batch_evaluate_json_files(
            args.model, 
            args.templated_json_files, 
            args.num_samples
        ))
    
    # Evaluate all JSON files in directory
    if args.json_dir:
        print(f"Batch processing directory: {args.json_dir}")
        
        # Use batch processing function from evaluate_commercial_api
        results_summary = batch_process_json_directory(
            args.model, 
            args.json_dir, 
            args.num_samples
        )
        
        # Save batch summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"batch_summary_{args.model}_{timestamp}.json"
        output_dir = "output/commercial_api"
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)
        
        # Print batch summary
        print(f"\n=== Batch Processing Result Summary ===")
        print(f"Directory: {args.json_dir}")
        print(f"Successfully processed: {results_summary['processed_files']} file(s)")
        print(f"Failed: {results_summary['failed_files']} file(s)")
        print(f"Summary saved to: {summary_path}")
        
        print(f"\nDetailed results:")
        for result in results_summary['results']:
            if result['status'] == 'success':
                print(f"✅ {result['file']}: jailbreak rate {result['jailbreak_success_rate']:.4f}, responses {result['total_responses']}")
            else:
                print(f"❌ {result['file']}: failed - {result.get('error', 'Unknown error')}")
        
        return  # Directory mode returns directly
    
    # Save batch evaluation summary
    if results:
        summary = {
            "model": args.model,
            "evaluated_templates": len(results),
            "total_samples": args.num_samples,
            "summary": [],
            "detailed_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        for result in results:
            summary["summary"].append({
                "template_id": result["template_id"],
                "template_name": result["template_name"],
                "jailbreak_success_rate": result["metrics"]["jailbreak_success_rate"],
                "total_responses": result["metrics"]["total_responses"]
            })
        
        if args.output_summary:
            summary_file = args.output_summary
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = f"batch_summary_{args.model}_{timestamp}.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\nBatch evaluation summary saved to: {summary_file}")
        
        # Print overall summary
        print(f"\n{'='*60}")
        print(f"Batch evaluation complete - model: {args.model}")
        print(f"{'='*60}")
        for item in summary["summary"]:
            print(f"{item['template_id']:20} | Jailbreak rate: {item['jailbreak_success_rate']:.4f} | Responses: {item['total_responses']}")

if __name__ == "__main__":
    main()
