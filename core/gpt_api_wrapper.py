"""
GPT-4 API Wrapper

用于通过 OpenAI API 调用 GPT-4 模型
"""

import os
from openai import OpenAI


class GPT4APIWrapper:
    """
    GPT-4 API 调用封装类
    
    提供与 HuggingFace pipeline 兼容的接口
    """
    
    def __init__(self, model_name="gpt-4", api_key=None):
        """
        初始化 GPT-4 API wrapper
        
        Args:
            model_name (str): 模型名称，默认 "gpt-4"
            api_key (str): OpenAI API密钥，如果为None则从环境变量OPENAI_API_KEY获取
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("必须提供api_key参数或设置OPENAI_API_KEY环境变量")
        
        self.client = OpenAI(api_key=self.api_key)
        self.is_api_model = True  # 标记这是API模型
    
    def __call__(self, prompts, max_new_tokens=256, **kwargs):
        """
        调用 GPT-4 生成响应
        
        Args:
            prompts (str or list): 单个prompt或prompt列表
            max_new_tokens (int): 最大生成token数
            **kwargs: 其他参数（保留以兼容HuggingFace接口）
        
        Returns:
            list: 格式与HuggingFace pipeline兼容的输出
        """
        # 处理单个prompt和批量prompts
        if isinstance(prompts, str):
            prompts = [prompts]
            single_input = True
        else:
            single_input = False
        
        results = []
        
        for prompt in prompts:
            try:
                # 调用 OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_new_tokens,
                    temperature=kwargs.get('temperature', 1.0)
                )
                
                # 提取生成的文本
                generated_text = response.choices[0].message.content
                
                # 组合完整输出（prompt + 生成内容）
                full_output = prompt + generated_text
                
                # 返回与HuggingFace pipeline兼容的格式
                results.append([{"generated_text": full_output}])
            
            except Exception as e:
                raise Exception(f"GPT-4 API调用失败: {e}")
        
        # 如果是单个输入，返回单个结果
        if single_input:
            return results[0]
        
        return results


def load_gpt4_model(model_name="gpt-4", api_key=None):
    """
    加载 GPT-4 模型（通过API）
    
    Args:
        model_name (str): 模型名称
        api_key (str): OpenAI API密钥
    
    Returns:
        GPT4APIWrapper: GPT-4 API wrapper对象
    """
    print(f"正在初始化 GPT-4 API 连接: {model_name}")
    
    try:
        wrapper = GPT4APIWrapper(model_name=model_name, api_key=api_key)
        print("✓ GPT-4 API 连接成功")
        return wrapper
    
    except Exception as e:
        raise Exception(f"GPT-4 API 初始化失败: {e}")


# 示例用法
if __name__ == "__main__":
    # 需要先设置环境变量 OPENAI_API_KEY
    # export OPENAI_API_KEY="your-openai-api-key-here"
    
    try:
        # 初始化 GPT-4
        gpt4 = load_gpt4_model()
        
        # 测试单个prompt
        test_prompt = "Hello, how are you?"
        print(f"\n测试prompt: {test_prompt}")
        
        result = gpt4(test_prompt, max_new_tokens=50)
        print(f"\n生成结果: {result[0]['generated_text']}")
        
        # 测试批量prompts
        batch_prompts = [
            "What is AI?",
            "Tell me a joke."
        ]
        print(f"\n测试批量prompts...")
        
        batch_results = gpt4(batch_prompts, max_new_tokens=50)
        for i, result in enumerate(batch_results):
            print(f"\nPrompt {i+1}: {batch_prompts[i]}")
            print(f"结果: {result[0]['generated_text']}")
    
    except ValueError as e:
        print(f"\n错误: {e}")
        print("请先设置OPENAI_API_KEY环境变量：")
        print("export OPENAI_API_KEY='your-api-key-here'")
    except Exception as e:
        print(f"\n错误: {e}")

