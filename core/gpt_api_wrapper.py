"""
GPT-4 API Wrapper

Call GPT-4 models via the OpenAI API.
"""

import os
from openai import OpenAI


class GPT4APIWrapper:
    """
    GPT-4 API wrapper class.
    
    Provides an interface compatible with HuggingFace pipeline.
    """
    
    def __init__(self, model_name="gpt-4", api_key=None):
        """
        Initialize GPT-4 API wrapper.
        
        Args:
            model_name (str): Model name, default "gpt-4"
            api_key (str): OpenAI API key; if None, reads OPENAI_API_KEY from environment
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("Must provide api_key or set OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        self.is_api_model = True  # Mark as API model
    
    def __call__(self, prompts, max_new_tokens=256, **kwargs):
        """
        Call GPT-4 to generate responses.
        
        Args:
            prompts (str or list): Single prompt or list of prompts
            max_new_tokens (int): Maximum number of tokens to generate
            **kwargs: Other parameters (kept for HuggingFace interface compatibility)
        
        Returns:
            list: Output format compatible with HuggingFace pipeline
        """
        # Handle single prompt and batch prompts
        if isinstance(prompts, str):
            prompts = [prompts]
            single_input = True
        else:
            single_input = False
        
        results = []
        
        for prompt in prompts:
            try:
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_new_tokens,
                    temperature=kwargs.get('temperature', 1.0)
                )
                
                # Extract generated text
                generated_text = response.choices[0].message.content
                
                # Combine full output (prompt + generated content)
                full_output = prompt + generated_text
                
                # Return format compatible with HuggingFace pipeline
                results.append([{"generated_text": full_output}])
            
            except Exception as e:
                raise Exception(f"GPT-4 API call failed: {e}")
        
        # Return single result for single input
        if single_input:
            return results[0]
        
        return results


def load_gpt4_model(model_name="gpt-4", api_key=None):
    """
    Load GPT-4 model (via API).
    
    Args:
        model_name (str): Model name
        api_key (str): OpenAI API key
    
    Returns:
        GPT4APIWrapper: GPT-4 API wrapper instance
    """
    print(f"Initializing GPT-4 API connection: {model_name}")
    
    try:
        wrapper = GPT4APIWrapper(model_name=model_name, api_key=api_key)
        print("✓ GPT-4 API connection successful")
        return wrapper
    
    except Exception as e:
        raise Exception(f"GPT-4 API initialization failed: {e}")


# Example usage
if __name__ == "__main__":
    # Set OPENAI_API_KEY first:
    # export OPENAI_API_KEY="your-openai-api-key-here"
    
    try:
        # Initialize GPT-4
        gpt4 = load_gpt4_model()
        
        # Test single prompt
        test_prompt = "Hello, how are you?"
        print(f"\nTest prompt: {test_prompt}")
        
        result = gpt4(test_prompt, max_new_tokens=50)
        print(f"\nGenerated result: {result[0]['generated_text']}")
        
        # Test batch prompts
        batch_prompts = [
            "What is AI?",
            "Tell me a joke."
        ]
        print(f"\nTesting batch prompts...")
        
        batch_results = gpt4(batch_prompts, max_new_tokens=50)
        for i, result in enumerate(batch_results):
            print(f"\nPrompt {i+1}: {batch_prompts[i]}")
            print(f"Result: {result[0]['generated_text']}")
    
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set OPENAI_API_KEY first:")
        print("export OPENAI_API_KEY='your-api-key-here'")
    except Exception as e:
        print(f"\nError: {e}")
