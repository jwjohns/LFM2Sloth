"""Core data processing functionality"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datasets import Dataset, load_dataset
import jsonlines


class DataProcessor:
    """Handles data loading, validation, and preprocessing for training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_format = config.get("format", "chatml")
        self.system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    
    def load_data(self, file_path: Union[str, Path], split: str = "train") -> Dataset:
        """Load data from various file formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if file_path.suffix == ".jsonl":
            return self._load_jsonl(file_path)
        elif file_path.suffix == ".csv":
            return self._load_csv(file_path)
        elif file_path.suffix == ".json":
            return self._load_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_jsonl(self, file_path: Path) -> Dataset:
        """Load JSONL format data"""
        data = []
        with jsonlines.open(file_path) as reader:
            for item in reader:
                data.append(item)
        return Dataset.from_list(data)
    
    def _load_csv(self, file_path: Path) -> Dataset:
        """Load CSV format data"""
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)
    
    def _load_json(self, file_path: Path) -> Dataset:
        """Load JSON format data"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return Dataset.from_list(data)
        elif isinstance(data, dict):
            return Dataset.from_dict(data)
        else:
            raise ValueError("JSON data must be a list or dictionary")
    
    def validate_data(self, dataset: Dataset) -> bool:
        """Validate dataset format and content"""
        required_fields = self._get_required_fields()
        
        for field in required_fields:
            if field not in dataset.column_names:
                raise ValueError(f"Required field '{field}' not found in dataset")
        
        # Check for empty messages
        if "messages" in dataset.column_names:
            for i, example in enumerate(dataset):
                if not example["messages"] or len(example["messages"]) == 0:
                    raise ValueError(f"Empty messages found at index {i}")
        
        return True
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields based on data format"""
        if self.data_format == "chatml":
            return ["messages"]
        elif self.data_format == "alpaca":
            return ["instruction", "output"]
        else:
            return []
    
    def preprocess_dataset(self, dataset: Dataset, tokenizer) -> Dataset:
        """Apply tokenization and formatting to dataset"""
        if self.data_format == "chatml":
            return dataset.map(
                lambda x: self._format_chatml(x, tokenizer),
                remove_columns=dataset.column_names
            )
        elif self.data_format == "alpaca":
            return dataset.map(
                lambda x: self._format_alpaca(x, tokenizer),
                remove_columns=dataset.column_names
            )
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")
    
    def _format_chatml(self, example: Dict[str, Any], tokenizer) -> Dict[str, str]:
        """Format data using ChatML template"""
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
    
    def _format_alpaca(self, example: Dict[str, Any], tokenizer) -> Dict[str, str]:
        """Format data using Alpaca template"""
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]
        
        # Create messages format for consistency
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{instruction}\n{input_text}".strip()},
            {"role": "assistant", "content": output}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False, 
            tokenize=False
        )
        
        return {"text": text}