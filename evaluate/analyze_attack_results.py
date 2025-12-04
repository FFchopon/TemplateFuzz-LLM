#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析单次攻击结果文件的脚本
处理JSON文件中的攻破问题数量和平均攻击次数
"""

import json
import sys
import os
import csv
from datetime import datetime

def analyze_attack_results(json_file_path):
    """
    分析攻击结果文件
    
    Args:
        json_file_path (str): JSON结果文件路径
        
    Returns:
        dict: 包含分析结果的字典
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取基本信息
        total_questions = data.get('attack_phase', {}).get('total_questions', 0)
        successful_attacks = data.get('successful_attacks', [])
        failed_attacks = data.get('failed_attacks', [])
        
        # 计算攻破问题数
        successful_count = len(successful_attacks)
        failed_count = len(failed_attacks)
        
        # 提取每个成功攻破问题的攻击次数
        attack_attempts = []
        for attack in successful_attacks:
            attempts = attack.get('attempts', 0)
            attack_attempts.append(attempts)
        
        # 计算平均攻击次数
        if attack_attempts:
            average_attempts = sum(attack_attempts) / len(attack_attempts)
        else:
            average_attempts = 0
        
        # 构建结果
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
        print("错误：找不到文件 {}".format(json_file_path))
        return None
    except json.JSONDecodeError:
        print("错误：无法解析JSON文件 {}".format(json_file_path))
        return None
    except Exception as e:
        print("错误：处理文件时发生异常 - {}".format(str(e)))
        return None

def print_results(results):
    """
    打印分析结果
    
    Args:
        results (dict): 分析结果字典
    """
    if results is None:
        return
    
    print("=" * 60)
    print("攻击结果分析报告")
    print("=" * 60)
    print("总问题数: {}".format(results['total_questions']))
    print("攻破问题数: {}/{}".format(results['successful_count'], results['total_questions']))
    print("失败问题数: {}".format(results['failed_count']))
    print("成功率: {:.2%}".format(results['success_rate']))
    print("-" * 40)
    print("平均攻击次数: {:.2f}".format(results['average_attempts']))
    print("最少攻击次数: {}".format(results['min_attempts']))
    print("最多攻击次数: {}".format(results['max_attempts']))
    print("-" * 40)
    
    # 显示攻击次数分布
    if results['attack_attempts']:
        print("攻击次数分布:")
        attempts_count = {}
        for attempts in results['attack_attempts']:
            attempts_count[attempts] = attempts_count.get(attempts, 0) + 1
        
        # 按攻击次数排序
        for attempts in sorted(attempts_count.keys()):
            count = attempts_count[attempts]
            percentage = count / len(results['attack_attempts']) * 100
            print("  {}次攻击: {}个问题 ({:.1f}%)".format(attempts, count, percentage))
    
    print("=" * 60)

def main():
    """主函数"""
    # 默认文件路径
    default_file = "output/Llama-2-7b-chat-hf/single_attack_bandit_strategy/single_attack_results/single_attack_Llama-2-7b-chat-hf_bandit_strategy_20250730_165251.json"
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        json_file_path = default_file
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print("错误：文件不存在 - {}".format(json_file_path))
        print("用法: python {} [JSON文件路径]".format(sys.argv[0]))
        print("默认文件: {}".format(default_file))
        sys.exit(1)
    
    print("正在分析文件: {}".format(json_file_path))
    print()
    
    # 分析结果
    results = analyze_attack_results(json_file_path)
    
    # 打印结果
    print_results(results)
    
    # 保存综合结果到CSV文件
    if results:
        base_name = json_file_path.replace('.json', '')
        output_file = "{}_results.csv".format(base_name)
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入总体统计信息
                writer.writerow(['攻击结果分析报告'])
                writer.writerow(['分析时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['源文件', os.path.basename(json_file_path)])
                writer.writerow([])  # 空行分隔
                
                # 总体统计
                writer.writerow(['总体统计'])
                writer.writerow(['指标', '数值'])
                writer.writerow(['总问题数', results['total_questions']])
                writer.writerow(['攻破问题数', results['successful_count']])
                writer.writerow(['失败问题数', results['failed_count']])
                writer.writerow(['成功率(%)', "{:.2f}".format(results['success_rate']*100)])
                writer.writerow(['平均攻击次数', "{:.2f}".format(results['average_attempts'])])
                writer.writerow(['最少攻击次数', results['min_attempts']])
                writer.writerow(['最多攻击次数', results['max_attempts']])
                writer.writerow([])  # 空行分隔
                
                # 详细攻击次数数据
                writer.writerow(['每个问题的攻击次数详情'])
                writer.writerow(['问题序号', '攻击次数', '攻击次数区间'])
                
                for i, attempts in enumerate(results['attack_attempts'], 1):
                    # 定义攻击次数区间
                    if attempts == 1:
                        interval = '1次'
                    elif attempts <= 5:
                        interval = '2-5次'
                    elif attempts <= 10:
                        interval = '6-10次'
                    elif attempts <= 20:
                        interval = '11-20次'
                    elif attempts <= 50:
                        interval = '21-50次'
                    else:
                        interval = '50次以上'
                    
                    writer.writerow([i, attempts, interval])
                
                writer.writerow([])  # 空行分隔
                
                # 攻击次数分布统计
                writer.writerow(['攻击次数分布统计'])
                writer.writerow(['攻击次数', '问题数量', '占比(%)'])
                
                attempts_count = {}
                for attempts in results['attack_attempts']:
                    attempts_count[attempts] = attempts_count.get(attempts, 0) + 1
                
                # 按攻击次数排序
                for attempts in sorted(attempts_count.keys()):
                    count = attempts_count[attempts]
                    percentage = count / len(results['attack_attempts']) * 100
                    writer.writerow([attempts, count, "{:.1f}".format(percentage)])
            
            print("\n综合分析结果已保存到: {}".format(output_file))
            
        except Exception as e:
            print("警告：无法保存CSV结果文件 - {}".format(str(e)))

if __name__ == "__main__":
    main()
