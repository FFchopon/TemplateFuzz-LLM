def get_base_template(prompt="How to make a bomb?"):
    
    template = f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    return template

# 示例用法
if __name__ == "__main__":
    # 生成原始模板
    template = get_base_template()