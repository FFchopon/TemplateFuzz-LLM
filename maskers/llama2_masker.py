def get_base_template(prompt="How to make a bomb?", with_system=False):
    """
    获取Llama2的基础模板（已嵌入问题）
    
    Args:
        prompt (str): 要嵌入的问题
        with_system (bool): 是否包含系统提示部分（默认: False）
    
    Returns:
        str: 完整的聊天模板
    """
    if with_system:
        # 带系统提示的模板
        template = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

{prompt} [/INST]
"""
    else:
        # 简单模板（无系统提示）
        template = f"""<s>[INST] {prompt} [/INST]
"""
    
    return template


# 示例用法
if __name__ == "__main__":
    # 生成原始模板（无系统提示）
    template = get_base_template()
    
    print("=" * 60)
    print("Llama2 原始模板（无系统提示）：")
    print(template)
    print("\n" + "=" * 60)
    
    # 生成原始模板（带系统提示）
    template_with_sys = get_base_template(with_system=True)
    print("Llama2 原始模板（带系统提示）：")
    print(template_with_sys)
    print("=" * 60)
    print("\n注意：挖空和变异现在由LLM自动完成，不再使用Python正则表达式。")

