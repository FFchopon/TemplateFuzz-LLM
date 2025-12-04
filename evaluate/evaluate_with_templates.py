#!/usr/bin/env python3
"""
商用大模型API评估框架 - 支持模板文件
"""

import json
import argparse
import os
from .evaluate_commercial_api import evaluate_template, evaluate_templated_json, batch_process_json_directory, save_results, MODEL_CONFIGS, load_template, is_chat_template
import logging
from datetime import datetime

def load_templates(template_file="templates.json"):
    """从JSON文件加载模板配置"""
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['templates']
    except Exception as e:
        logging.error(f"加载模板文件失败: {e}")
        return {}

def list_templates(templates):
    """列出所有可用模板"""
    print("可用模板:")
    print("-" * 50)
    for key, template_info in templates.items():
        print(f"ID: {key}")
        print(f"名称: {template_info['name']}")
        print(f"描述: {template_info['description']}")
        print(f"模板: {template_info['template'][:100]}...")
        print("-" * 50)

def batch_evaluate(model_name, template_ids, templates, num_samples=None):
    """批量评估多个模板"""
    results = []
    
    for template_id in template_ids:
        if template_id not in templates:
            logging.warning(f"模板 '{template_id}' 不存在，跳过")
            continue
        
        template_info = templates[template_id]
        template_str = template_info['template']
        template_name = template_info['name']
        
        print(f"\n{'='*60}")
        print(f"评估模板: {template_id} ({template_name})")
        print(f"{'='*60}")
        print(f"模板内容: {template_str}")
        print()
        
        try:
            jailbreak_rate, experiment_results = evaluate_template(
                model_name, template_str, num_samples, False  # templates.json中的模板不是文件模板
            )
            
            # 准备结果
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
            
            # 保存单个模板结果
            save_results(result, model_name, template_id)
            
            # 打印摘要
            print(f"\n--- {template_id} 结果摘要 ---")
            print(f"越狱成功率: {jailbreak_rate:.4f}")
            print(f"总回答数: {result['metrics']['total_responses']}")
            
        except Exception as e:
            logging.error(f"评估模板 '{template_id}' 时发生错误: {e}")
            continue
    
    return results

def batch_evaluate_txt_files(model_name, template_files, num_samples=None):
    """批量评估.txt模板文件"""
    results = []
    
    for template_file in template_files:
        if not template_file.endswith('.txt'):
            logging.warning(f"跳过非.txt文件: {template_file}")
            continue
        
        try:
            # 加载模板文件
            template_str, is_file_template = load_template(template_file)
            template_name = os.path.basename(template_file).replace('.txt', '')
            
            print(f"\n{'='*60}")
            print(f"评估模板文件: {template_file}")
            print(f"{'='*60}")
            if is_chat_template(template_str):
                print("检测到Jinja2 chat template格式")
            print(f"模板内容预览: {template_str[:200]}...")
            print()
            
            jailbreak_rate, experiment_results = evaluate_template(
                model_name, template_str, num_samples, is_file_template
            )
            
            # 准备结果
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
            
            # 保存单个模板结果
            save_results(result, model_name, template_name)
            
            # 打印摘要
            print(f"\n--- {template_name} 结果摘要 ---")
            print(f"越狱成功率: {jailbreak_rate:.4f}")
            print(f"总回答数: {result['metrics']['total_responses']}")
            
        except Exception as e:
            logging.error(f"评估模板文件 '{template_file}' 时发生错误: {e}")
            continue
    
    return results

def batch_evaluate_json_files(model_name, json_files, num_samples=None):
    """批量评估已处理的JSON文件"""
    results = []
    
    for json_file in json_files:
        if not json_file.endswith('.json'):
            logging.warning(f"跳过非.json文件: {json_file}")
            continue
        
        try:
            template_name = os.path.basename(json_file).replace('.json', '')
            
            print(f"\n{'='*60}")
            print(f"评估JSON文件: {json_file}")
            print(f"{'='*60}")
            
            jailbreak_rate, experiment_results = evaluate_templated_json(
                model_name, json_file, num_samples
            )
            
            # 准备结果
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
            
            # 保存单个JSON文件结果
            save_results(result, model_name, template_name)
            
            # 打印摘要
            print(f"\n--- {template_name} 结果摘要 ---")
            print(f"越狱成功率: {jailbreak_rate:.4f}")
            print(f"总回答数: {result['metrics']['total_responses']}")
            
        except Exception as e:
            logging.error(f"评估JSON文件 '{json_file}' 时发生错误: {e}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="商用大模型API评估框架 - 支持模板文件")
    parser.add_argument("--model", required=True, 
                       choices=list(MODEL_CONFIGS.keys()),
                       help="模型名称")
    parser.add_argument("--template-file", default="templates.json",
                       help="模板配置文件路径")
    parser.add_argument("--template-ids", nargs="+",
                       help="要评估的模板ID列表")
    parser.add_argument("--template-txt-files", nargs="+",
                       help="直接指定.txt模板文件路径列表")
    parser.add_argument("--templated-json-files", nargs="+",
                       help="直接指定包含已处理templated_input的JSON文件路径列表")
    parser.add_argument("--json-dir", 
                       help="包含多个JSON文件的目录路径，批量处理（仅处理越狱提示词）")
    parser.add_argument("--list-templates", action="store_true",
                       help="列出所有可用模板")
    parser.add_argument("--num-samples", type=int,
                       help="样本数量，不指定则使用全部数据")
    parser.add_argument("--output-summary", 
                       help="保存批量评估摘要的文件名")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 加载模板
    templates = load_templates(args.template_file)
    if not templates:
        print("无法加载模板文件")
        return
    
    # 列出模板
    if args.list_templates:
        list_templates(templates)
        return
    
    # 检查模板参数
    if not args.template_ids and not args.template_txt_files and not args.templated_json_files and not args.json_dir:
        print("请指定要评估的模板ID（--template-ids）、模板文件（--template-txt-files）、JSON文件（--templated-json-files）或目录（--json-dir），或使用 --list-templates 查看可用模板")
        return
    
    print(f"开始批量评估模型: {args.model}")
    
    results = []
    
    # 评估JSON模板文件中的模板
    if args.template_ids:
        print(f"评估JSON模板: {args.template_ids}")
        results.extend(batch_evaluate(
            args.model, 
            args.template_ids, 
            templates, 
            args.num_samples
        ))
    
    # 评估.txt模板文件
    if args.template_txt_files:
        print(f"评估.txt模板文件: {args.template_txt_files}")
        results.extend(batch_evaluate_txt_files(
            args.model, 
            args.template_txt_files, 
            args.num_samples
        ))
    
    # 评估已处理的JSON文件
    if args.templated_json_files:
        print(f"评估已处理的JSON文件: {args.templated_json_files}")
        results.extend(batch_evaluate_json_files(
            args.model, 
            args.templated_json_files, 
            args.num_samples
        ))
    
    # 评估目录中的所有JSON文件
    if args.json_dir:
        print(f"批量处理目录: {args.json_dir}")
        
        # 直接使用evaluate_commercial_api中的批量处理函数
        results_summary = batch_process_json_directory(
            args.model, 
            args.json_dir, 
            args.num_samples
        )
        
        # 保存批量摘要
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"batch_summary_{args.model}_{timestamp}.json"
        output_dir = "output/commercial_api"
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)
        
        # 打印批量摘要
        print(f"\n=== 批量处理结果摘要 ===")
        print(f"目录: {args.json_dir}")
        print(f"处理成功: {results_summary['processed_files']} 个文件")
        print(f"处理失败: {results_summary['failed_files']} 个文件")
        print(f"摘要已保存到: {summary_path}")
        
        print(f"\n详细结果:")
        for result in results_summary['results']:
            if result['status'] == 'success':
                print(f"✅ {result['file']}: 越狱率 {result['jailbreak_success_rate']:.4f}, 回答数 {result['total_responses']}")
            else:
                print(f"❌ {result['file']}: 失败 - {result.get('error', 'Unknown error')}")
        
        return  # 目录模式直接返回，不继续后续处理
    
    # 保存批量评估摘要
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
        
        print(f"\n批量评估摘要已保存到: {summary_file}")
        
        # 打印总体摘要
        print(f"\n{'='*60}")
        print(f"批量评估完成 - 模型: {args.model}")
        print(f"{'='*60}")
        for item in summary["summary"]:
            print(f"{item['template_id']:20} | 越狱率: {item['jailbreak_success_rate']:.4f} | 回答数: {item['total_responses']}")

if __name__ == "__main__":
    main() 
