"""Data format templates and formatters"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseFormatter(ABC):
    """Base class for data formatters"""
    
    @abstractmethod
    def format_example(self, example: Dict[str, Any], tokenizer) -> Dict[str, str]:
        """Format a single example for training"""
        pass
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Get required fields for this format"""
        pass


class ChatMLFormatter(BaseFormatter):
    """Formatter for ChatML conversation format"""
    
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt
    
    def format_example(self, example: Dict[str, Any], tokenizer) -> Dict[str, str]:
        """Format example using ChatML template"""
        messages = example["messages"]
        
        # Ensure system message exists
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False
        )
        
        return {"text": text}
    
    def get_required_fields(self) -> List[str]:
        return ["messages"]


class AlpacaFormatter(BaseFormatter):
    """Formatter for Alpaca instruction format"""
    
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt
    
    def format_example(self, example: Dict[str, Any], tokenizer) -> Dict[str, str]:
        """Format example using Alpaca template"""
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]
        
        # Convert to messages format
        user_content = f"{instruction}\n{input_text}".strip() if input_text else instruction
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False
        )
        
        return {"text": text}
    
    def get_required_fields(self) -> List[str]:
        return ["instruction", "output"]


class CustomerServiceFormatter(ChatMLFormatter):
    """Specialized formatter for customer service conversations"""
    
    def __init__(self, company_name: str = "ACME"):
        system_prompt = f"You are a helpful, empathetic customer support agent for {company_name}. Provide concise, professional assistance while being understanding of customer concerns."
        super().__init__(system_prompt)


class CodeAssistantFormatter(ChatMLFormatter):
    """Specialized formatter for code assistance conversations"""
    
    def __init__(self):
        system_prompt = "You are a helpful programming assistant. Provide clear, well-commented code examples and explanations."
        super().__init__(system_prompt)