"""Main training orchestration"""

import os
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainerCallback, TrainerState, TrainerControl
import logging

from ..data import DataProcessor
from ..model import ModelLoader, ModelConfig, LoRAConfig
from .config import TrainingConfig
from .tracker import create_tracker, ExperimentTracker


logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainerCallback):
    """Custom early stopping callback for SFT training"""
    
    def __init__(self, early_stopping_patience: int = 5, early_stopping_threshold: float = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.patience_counter = 0
        self.best_metric = None
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, model, **kwargs):
        current_metric = state.log_history[-1].get("eval_loss")
        
        if current_metric is None:
            return
        
        if self.best_metric is None or current_metric < self.best_metric - self.early_stopping_threshold:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.early_stopping_patience:
            logger.info(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
            control.should_training_stop = True


class ExperimentTrackingCallback(TrainerCallback):
    """Callback for experiment tracking integration"""
    
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self.start_time = None
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training"""
        self.start_time = time.time()
        if self.tracker:
            self.tracker.log_text(f"Training started at step {state.global_step}", "training_log")
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        """Called when logging metrics"""
        if self.tracker and logs:
            # Filter out non-numeric values
            numeric_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if numeric_logs:
                self.tracker.log_metrics(numeric_logs, step=state.global_step)
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called after evaluation"""
        if self.tracker:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            self.tracker.log_metrics({
                "epoch": state.epoch,
                "training_time_minutes": elapsed_time / 60
            }, step=state.global_step)
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training"""
        if self.tracker:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            self.tracker.log_metrics({
                "total_training_time_minutes": elapsed_time / 60,
                "final_global_step": state.global_step
            })
            self.tracker.log_text(f"Training completed at step {state.global_step}", "training_log")


class Trainer:
    """Main training orchestrator for the modular pipeline"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        lora_config: Optional[LoRAConfig] = None,
        data_config: Optional[Dict[str, Any]] = None
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.lora_config = lora_config
        self.data_config = data_config or {}
        
        # Initialize components
        self.model_loader = ModelLoader(model_config)
        self.data_processor = DataProcessor(self.data_config)
        
        # Training state
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Initialize experiment tracking
        self.tracker = create_tracker(training_config)
        self.training_stats = {}
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for training"""
        log_level = logging.INFO
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=log_level,
        )
    
    def prepare_model(self) -> Tuple[Any, Any]:
        """Load and prepare model and tokenizer"""
        logger.info(f"Loading model: {self.model_config.model_name}")
        
        start_time = time.time()
        
        # Load with Unsloth for optimal training
        self.model, self.tokenizer = self.model_loader.load_with_unsloth(self.lora_config)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Log model info
        model_info = self.model_loader.get_model_info()
        for key, value in model_info.items():
            logger.info(f"{key}: {value}")
        
        return self.model, self.tokenizer
    
    def prepare_datasets(self, train_path: str, eval_path: Optional[str] = None) -> Tuple[Dataset, Optional[Dataset]]:
        """Load and prepare training datasets"""
        logger.info(f"Loading training data: {train_path}")
        
        # Load datasets
        train_dataset = self.data_processor.load_data(train_path)
        eval_dataset = None
        
        if eval_path:
            logger.info(f"Loading evaluation data: {eval_path}")
            eval_dataset = self.data_processor.load_data(eval_path)
        
        # Validate data format
        self.data_processor.validate_data(train_dataset)
        if eval_dataset:
            self.data_processor.validate_data(eval_dataset)
        
        # Preprocess datasets
        if self.tokenizer is None:
            raise ValueError("Model must be prepared before datasets")
        
        train_dataset = self.data_processor.preprocess_dataset(train_dataset, self.tokenizer)
        
        if eval_dataset:
            eval_dataset = self.data_processor.preprocess_dataset(eval_dataset, self.tokenizer)
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute training process"""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be prepared before training")
        
        logger.info("Starting training...")
        
        # Prepare training arguments
        training_args = self.training_config.get_training_args()
        sft_args = self.training_config.get_sft_args()
        
        # Initialize experiment tracking
        if self.tracker:
            try:
                # Prepare config for tracking
                tracking_config = self.training_config.to_dict()
                tracking_config.update(self.model_config.to_dict())
                if self.lora_config:
                    tracking_config.update(self.lora_config.to_dict())
                
                # Initialize tracker
                self.tracker.initialize(tracking_config)
                logger.info(f"Initialized experiment tracking: {self.training_config.tracking_provider}")
            except Exception as e:
                logger.error(f"Failed to initialize tracking: {e}")
                self.tracker = None
        
        # Set up callbacks
        callbacks = []
        if self.training_config.early_stopping_patience:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.training_config.early_stopping_patience,
                early_stopping_threshold=self.training_config.early_stopping_threshold or 0.0
            ))
        
        # Add tracking callback
        if self.tracker:
            callbacks.append(ExperimentTrackingCallback(self.tracker))
        
        # Create SFT trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            **sft_args
        )
        
        # Track training start time
        start_time = time.time()
        
        # Start training
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Gather training statistics
        self.training_stats = {
            "training_time": training_time,
            "total_steps": train_result.global_step,
            "training_loss": train_result.training_loss,
            "epochs_completed": train_result.global_step / self.training_config.estimate_steps_per_epoch(len(train_dataset)),
        }
        
        # Add evaluation metrics if available
        if self.trainer.state.log_history:
            last_log = self.trainer.state.log_history[-1]
            if "eval_loss" in last_log:
                self.training_stats["final_eval_loss"] = last_log["eval_loss"]
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Finish experiment tracking
        if self.tracker:
            try:
                self.tracker.finish()
                logger.info("Finished experiment tracking")
            except Exception as e:
                logger.error(f"Failed to finish experiment tracking: {e}")
        
        return self.training_stats
    
    def save_model(self, output_dir: Optional[str] = None, save_merged: bool = False) -> str:
        """Save the trained model"""
        
        if self.trainer is None:
            raise ValueError("No trained model to save")
        
        save_dir = output_dir or self.training_config.output_dir
        adapter_dir = os.path.join(save_dir, "adapter")
        
        # Save LoRA adapter
        logger.info(f"Saving LoRA adapter to: {adapter_dir}")
        self.trainer.save_model(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)
        
        # Save merged model if requested
        if save_merged:
            from unsloth import FastLanguageModel
            
            merged_dir = os.path.join(save_dir, "merged")
            logger.info(f"Saving merged model to: {merged_dir}")
            
            # Prepare for inference and save
            FastLanguageModel.for_inference(self.model)
            self.model.save_pretrained(merged_dir)
            self.tokenizer.save_pretrained(merged_dir)
        
        # Save training config and stats
        config_path = os.path.join(save_dir, "training_config.json")
        self.training_config.save_to_json(config_path)
        
        if self.training_stats:
            import json
            stats_path = os.path.join(save_dir, "training_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
        
        # Log model artifacts to experiment tracker
        if self.tracker and self.training_config.log_model_artifacts:
            try:
                if save_merged and os.path.exists(merged_dir):
                    self.tracker.log_model_artifact(merged_dir, "merged_model")
                self.tracker.log_model_artifact(adapter_dir, "lora_adapter")
                logger.info("Logged model artifacts to experiment tracker")
            except Exception as e:
                logger.error(f"Failed to log model artifacts: {e}")
        
        return save_dir
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Run evaluation on the model"""
        
        if self.trainer is None:
            raise ValueError("No trained model to evaluate")
        
        logger.info("Running evaluation...")
        
        eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        logger.info("Evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return eval_results
    
    def get_memory_usage(self) -> Dict[str, str]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {"status": "CUDA not available"}
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            "allocated": f"{memory_allocated:.2f} GB",
            "reserved": f"{memory_reserved:.2f} GB", 
            "total": f"{memory_total:.2f} GB",
            "utilization": f"{(memory_reserved/memory_total)*100:.1f}%"
        }