import re
import time
from typing import Any, List, Mapping, Optional, Tuple

import torch
from transformers import AutoTokenizer, pipeline
import openai
from openai import OpenAI

from post_process import RewardFunctionConverter


class HuggingFaceLLM:
    """Custom HuggingFace LLM wrapper without LangChain dependencies."""
    
    def __init__(self, name: str, temperature: float = 0, **kwargs):
        self.name = name
        self.temperature = temperature
        self.kwargs = kwargs
        
        # Model name mapping
        self.name_map = {
            "codellama_34b": "codellama/CodeLlama-34b-Instruct-hf",
            "llama_2_70b": "meta-llama/Llama-2-70b-chat-hf"
        }
        
        if self.name not in self.name_map:
            raise ValueError(f"Model name {self.name} not supported!")
            
        self.model_name = self.name_map[self.name]
        self._setup_model()
    
    def _setup_model(self):
        """Initialize the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.pipe.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        chat = [
            {"role": "user", "content": prompt},
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        
        raw_results = self.pipe(
            [formatted_prompt],
            do_sample=False,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=4096,
            batch_size=1
        )
        
        generated_text = raw_results[0][0]["generated_text"][len(formatted_prompt):]
        print(generated_text)
        return generated_text


class OpenAIChat:
    """Custom OpenAI Chat wrapper without LangChain dependencies."""
    
    def __init__(self, model_name: str = "gpt-5-mini", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.client = OpenAI(**kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

class genaiChat:
    """Custom genai Chat wrapper without LangChain dependencies."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.client = genai.Client(**kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt using genai API."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            **kwargs
        )
        return response.text

class ZeroShotGenerator:
    """Reward function generator without LangChain dependencies."""
    
    def __init__(self, info_prompt_template: str, model_name="gpt-5-mini", **kwargs) -> None:
        self.info_prompt_template = info_prompt_template
        
        if model_name in ["gpt-5", "gpt-3.5-turbo", "gpt-4", "gpt-4-0314", "gpt-5-mini"]: # require update
            self.llm = OpenAIChat(model_name=model_name, **kwargs)
        elif model_name in ["codellama_34b", "llama_2_70b"]:
            self.llm = HuggingFaceLLM(name=model_name, **kwargs)
        elif model_name in ["gemini-2.5-flash"]:
            self.llm = genaiChat(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Model name {model_name} not supported!")

    def generate_code(self, instruction: str, map_dict: dict) -> Tuple[str, str]:
        """Generate reward function code from instruction."""
        code_content = ""
        
        while True:
            # Format the prompt template with the instruction
            formatted_prompt = self.info_prompt_template.format(instruction=instruction)
            
            # Generate response using the LLM
            response = self.llm.generate(formatted_prompt)
            
            # Extract code from response using regex
            pattern = r"\```python\n(.+?)\n```" if "```python" in response else r"\```\n(.+?)\n```"
            match = re.search(pattern, response, re.DOTALL)
            
            if match:
                code_content = match.group(1)
                break
            else:
                print(response)
                time.sleep(5)
                print("No match!")
                continue

        general_code = code_content

        # Post-processing, replace the general terms with specific terms
        converter = RewardFunctionConverter(map_dict)
        specific_code = converter.general_to_specific(general_code)

        return general_code, specific_code