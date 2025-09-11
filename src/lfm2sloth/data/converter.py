"""Data format conversion utilities"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import jsonlines


class FormatConverter:
    """Convert between different data formats for training"""
    
    @staticmethod
    def csv_to_jsonl(
        csv_path: Path, 
        output_path: Path,
        user_column: str = "user_prompt",
        assistant_column: str = "agent_reply",
        system_prompt: str = "You are a helpful assistant."
    ) -> None:
        """Convert CSV to JSONL ChatML format"""
        
        df = pd.read_csv(csv_path)
        
        with jsonlines.open(output_path, mode='w') as writer:
            for _, row in df.iterrows():
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(row[user_column]).strip()},
                    {"role": "assistant", "content": str(row[assistant_column]).strip()}
                ]
                writer.write({"messages": messages})
    
    @staticmethod
    def json_to_jsonl(json_path: Path, output_path: Path) -> None:
        """Convert JSON array to JSONL format"""
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of objects")
        
        with jsonlines.open(output_path, mode='w') as writer:
            for item in data:
                writer.write(item)
    
    @staticmethod
    def alpaca_to_chatml(
        input_path: Path,
        output_path: Path,
        system_prompt: str = "You are a helpful assistant."
    ) -> None:
        """Convert Alpaca format to ChatML format"""
        
        # Load data
        if input_path.suffix == ".jsonl":
            with jsonlines.open(input_path) as reader:
                data = list(reader)
        elif input_path.suffix == ".json":
            with open(input_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError("Input file must be JSON or JSONL")
        
        # Convert format
        converted_data = []
        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            user_content = f"{instruction}\n{input_text}".strip() if input_text else instruction
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
            
            converted_data.append({"messages": messages})
        
        # Save converted data
        with jsonlines.open(output_path, mode='w') as writer:
            for item in converted_data:
                writer.write(item)
    
    @staticmethod
    def validate_chatml_format(file_path: Path) -> bool:
        """Validate ChatML format JSONL file"""
        
        try:
            with jsonlines.open(file_path) as reader:
                for line_num, item in enumerate(reader, 1):
                    if "messages" not in item:
                        print(f"Line {line_num}: Missing 'messages' field")
                        return False
                    
                    messages = item["messages"]
                    if not isinstance(messages, list) or len(messages) == 0:
                        print(f"Line {line_num}: 'messages' must be a non-empty list")
                        return False
                    
                    for msg_idx, message in enumerate(messages):
                        if not isinstance(message, dict):
                            print(f"Line {line_num}, message {msg_idx}: Message must be a dict")
                            return False
                        
                        if "role" not in message or "content" not in message:
                            print(f"Line {line_num}, message {msg_idx}: Missing 'role' or 'content'")
                            return False
                        
                        if message["role"] not in ["system", "user", "assistant"]:
                            print(f"Line {line_num}, message {msg_idx}: Invalid role '{message['role']}'")
                            return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    @staticmethod
    def split_dataset(
        input_path: Path,
        train_path: Path,
        val_path: Path,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> None:
        """Split dataset into train and validation sets"""
        
        # Load data
        with jsonlines.open(input_path) as reader:
            data = list(reader)
        
        # Shuffle and split
        import random
        random.seed(seed)
        random.shuffle(data)
        
        split_idx = int(len(data) * (1 - val_ratio))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Save splits
        with jsonlines.open(train_path, mode='w') as writer:
            for item in train_data:
                writer.write(item)
        
        with jsonlines.open(val_path, mode='w') as writer:
            for item in val_data:
                writer.write(item)
        
        print(f"Split complete: {len(train_data)} train, {len(val_data)} validation samples")