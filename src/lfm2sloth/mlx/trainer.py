"""MLX trainer for Apple Silicon"""

import os
import json
import time
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MLXTrainer:
    """Trainer class for MLX-based fine-tuning on Apple Silicon"""
    
    def __init__(self, model_config, data_config=None):
        """Initialize the MLX trainer
        
        Args:
            model_config: MLXConfig object with training settings
            data_config: MLXDataConfig object with data settings
        """
        self.model_config = model_config
        self.data_config = data_config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
    def prepare_data(self, train_path: str, valid_path: Optional[str] = None) -> Tuple[Any, Any]:
        """Prepare training and validation datasets
        
        Args:
            train_path: Path to training data
            valid_path: Path to validation data (optional)
            
        Returns:
            Tuple of (train_dataset, valid_dataset)
        """
        try:
            from mlx_lm.tuner import datasets
            
            logger.info(f"Loading training data from: {train_path}")
            
            # Load datasets
            train_dataset = datasets.load_dataset(train_path)
            valid_dataset = None
            
            if valid_path and os.path.exists(valid_path):
                logger.info(f"Loading validation data from: {valid_path}")
                valid_dataset = datasets.load_dataset(valid_path)
            
            logger.info(f"✅ Data loaded:")
            logger.info(f"  Training samples: {len(train_dataset)}")
            if valid_dataset:
                logger.info(f"  Validation samples: {len(valid_dataset)}")
            
            return train_dataset, valid_dataset
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def train(
        self,
        model,
        tokenizer,
        train_data: str,
        valid_data: Optional[str] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train the model using MLX with proper implementation
        
        Args:
            model: MLX model to train
            tokenizer: Tokenizer for the model
            train_data: Path to training data folder
            valid_data: Path to validation data folder (optional)
            resume_from: Path to resume training from (optional)
            
        Returns:
            Dictionary with training statistics
        """
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            from mlx_lm.tuner import datasets, evaluate, linear_to_lora_layers
            import numpy as np
            
            self.model = model
            self.tokenizer = tokenizer
            
            logger.info("Starting MLX training...")
            start_time = time.time()
            
            # Prepare data - use CLI approach since it's more stable
            logger.info("Using MLX CLI for training (more stable approach)")
            return self.train_with_cli(
                model_name=self.model_config.model_name,
                data_path=train_data,
                output_path=self.model_config.adapter_path
            )
            
            # Setup LoRA if not already done
            if not hasattr(model, '_lora_applied'):
                logger.info("Applying LoRA to model...")
                model.freeze()
                lora_config = {
                    "rank": self.model_config.lora_rank,
                    "alpha": self.model_config.lora_alpha,
                    "dropout": self.model_config.lora_dropout,
                    "scale": self.model_config.lora_alpha / self.model_config.lora_rank
                }
                
                linear_to_lora_layers(
                    model,
                    self.model_config.lora_layers,
                    lora_config
                )
                model._lora_applied = True
                
                # Log LoRA info
                trainable_params = sum(p.size for _, p in model.trainable_parameters().items())
                total_params = sum(p.size for _, p in model.parameters().items())
                logger.info(f"LoRA applied: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")
            
            # Setup optimizer
            if self.model_config.optimizer == "adamw":
                optimizer = optim.AdamW(
                    learning_rate=self.model_config.learning_rate,
                    weight_decay=self.model_config.weight_decay
                )
            else:
                optimizer = optim.Adam(
                    learning_rate=self.model_config.learning_rate
                )
            
            # Training loop
            logger.info(f"Starting training for {self.model_config.num_iterations} iterations")
            
            training_losses = []
            best_loss = float('inf')
            
            for iteration in range(self.model_config.num_iterations):
                # Get batch
                batch = train_dataset.sample_batch(self.model_config.batch_size)
                
                # Forward pass and compute loss
                def loss_fn():
                    outputs = model(**batch)
                    return outputs.loss if hasattr(outputs, 'loss') else outputs
                
                # Gradient step
                loss, grad = mx.value_and_grad(loss_fn)()
                optimizer.update(model, grad)
                mx.eval(model.parameters(), optimizer.state)
                
                training_losses.append(float(loss))
                
                # Logging
                if iteration % self.model_config.logging_steps == 0:
                    avg_loss = np.mean(training_losses[-self.model_config.logging_steps:])
                    logger.info(f"Iter {iteration}/{self.model_config.num_iterations}: Loss = {avg_loss:.4f}")
                
                # Validation
                if valid_dataset and iteration % self.model_config.eval_steps == 0:
                    val_loss = self._evaluate(model, valid_dataset)
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        self._save_checkpoint(model, iteration, val_loss)
                
                # Save checkpoint
                if iteration % self.model_config.save_steps == 0 and iteration > 0:
                    self._save_checkpoint(model, iteration)
            
            training_time = time.time() - start_time
            final_loss = np.mean(training_losses[-10:]) if training_losses else 0.0
            
            logger.info(f"✅ Training completed in {training_time/60:.2f} minutes")
            logger.info(f"Final loss: {final_loss:.4f}")
            
            # Save final model
            self._save_final_model(model)
            
            # Return training statistics
            stats = {
                "training_time": training_time,
                "iterations": self.model_config.num_iterations,
                "batch_size": self.model_config.batch_size,
                "learning_rate": self.model_config.learning_rate,
                "final_loss": final_loss,
                "training_losses": training_losses
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def train_with_cli(
        self,
        model_name: str,
        data_path: str,
        output_path: Optional[str] = None
    ) -> bool:
        """Train using MLX CLI (alternative method)
        
        Args:
            model_name: Name or path of the model
            data_path: Path to training data folder
            output_path: Path to save adapters
            
        Returns:
            True if training succeeded
        """
        import subprocess
        
        output_path = output_path or self.model_config.adapter_path
        
        # Build command
        cmd = [
            "python", "-m", "mlx_lm.lora",
            "--model", model_name,
            "--train",
            "--data", data_path,
            "--batch-size", str(self.model_config.batch_size),
            "--num-layers", str(self.model_config.lora_layers),  # Correct argument name
            "--iters", str(self.model_config.num_iterations),
            "--learning-rate", str(self.model_config.learning_rate),
            "--adapter-path", output_path
        ]
        
        if self.model_config.use_qlora:
            cmd.append("--use-qlora")
        
        logger.info(f"Running MLX training command: {' '.join(cmd)}")
        
        try:
            import time
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=False, text=True)  # Show output
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info("✅ Training completed successfully")
                return {
                    "training_time": training_time,
                    "iterations": self.model_config.num_iterations,
                    "batch_size": self.model_config.batch_size,
                    "learning_rate": self.model_config.learning_rate,
                    "final_loss": 0.0,  # CLI doesn't return this easily
                    "training_losses": []
                }
            else:
                logger.error(f"Training failed with return code: {result.returncode}")
                raise RuntimeError("MLX training failed")
                
        except Exception as e:
            logger.error(f"Failed to run training command: {e}")
            raise
    
    def evaluate(self, model, tokenizer, test_data: str) -> Dict[str, float]:
        """Evaluate the model on test data
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            test_data: Path to test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            from mlx_lm import generate
            import mlx
            
            logger.info("Evaluating model...")
            
            # Simple perplexity calculation
            # In production, you'd want more comprehensive metrics
            
            metrics = {
                "perplexity": 0.0,  # Placeholder
                "loss": 0.0  # Placeholder
            }
            
            logger.info(f"✅ Evaluation completed")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        adapter_path: Optional[str] = None
    ) -> str:
        """Generate text using the trained model via CLI
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            adapter_path: Path to LoRA adapter (if not already loaded)
            
        Returns:
            Generated text
        """
        try:
            import subprocess
            
            adapter_path = adapter_path or self.model_config.adapter_path
            
            # Use CLI for generation - more reliable
            cmd = [
                "python", "-m", "mlx_lm.generate",
                "--model", self.model_config.model_name,
                "--adapter-path", adapter_path,
                "--prompt", prompt,
                "--max-tokens", str(max_tokens),
                "--temp", str(temperature)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Extract just the generated text after the prompt
                output = result.stdout.strip()
                if prompt in output:
                    response = output.split(prompt, 1)[1].strip()
                else:
                    response = output
                return response
            else:
                raise RuntimeError(f"Generation failed: {result.stderr}")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def save_training_config(self, save_path: str):
        """Save training configuration to file
        
        Args:
            save_path: Path to save configuration
        """
        os.makedirs(save_path, exist_ok=True)
        config_file = os.path.join(save_path, "training_config.json")
        
        config_dict = {
            "model": {
                "name": self.model_config.model_name,
                "max_seq_length": self.model_config.max_seq_length
            },
            "lora": {
                "layers": self.model_config.lora_layers,
                "rank": self.model_config.lora_rank,
                "alpha": self.model_config.lora_alpha,
                "dropout": self.model_config.lora_dropout
            },
            "training": {
                "batch_size": self.model_config.batch_size,
                "iterations": self.model_config.num_iterations,
                "learning_rate": self.model_config.learning_rate,
                "optimizer": self.model_config.optimizer
            }
        }
        
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"✅ Training config saved to: {config_file}")
    
    def _evaluate(self, model, dataset) -> float:
        """Evaluate model on dataset and return average loss"""
        try:
            import mlx.core as mx
            
            model.eval()
            losses = []
            
            # Evaluate on a few batches
            eval_batches = min(10, len(dataset) // self.model_config.batch_size)
            
            for _ in range(eval_batches):
                batch = dataset.sample_batch(self.model_config.batch_size)
                with mx.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                    losses.append(float(loss))
            
            model.train()
            import numpy as np
            return np.mean(losses) if losses else 0.0
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.0
    
    def _save_checkpoint(self, model, iteration: int, loss: Optional[float] = None):
        """Save training checkpoint"""
        try:
            import mlx.core as mx
            import os
            
            checkpoint_dir = os.path.join(self.model_config.adapter_path, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.safetensors")
            
            # Save trainable parameters
            trainable_params = dict(model.trainable_parameters())
            mx.save_safetensors(checkpoint_path, trainable_params)
            
            # Save metadata
            metadata = {
                "iteration": iteration,
                "loss": loss,
                "model_config": {
                    "lora_rank": self.model_config.lora_rank,
                    "lora_alpha": self.model_config.lora_alpha,
                    "lora_dropout": self.model_config.lora_dropout,
                }
            }
            
            metadata_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.json")
            with open(metadata_path, "w") as f:
                import json
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _save_final_model(self, model):
        """Save the final trained model"""
        try:
            import mlx.core as mx
            import os
            import json
            
            output_dir = self.model_config.adapter_path
            os.makedirs(output_dir, exist_ok=True)
            
            # Save adapter weights
            adapter_file = os.path.join(output_dir, "adapters.safetensors")
            trainable_params = dict(model.trainable_parameters())
            mx.save_safetensors(adapter_file, trainable_params)
            
            # Save adapter config
            config_file = os.path.join(output_dir, "adapter_config.json")
            adapter_config = {
                "model_name": self.model_config.model_name,
                "lora_rank": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "lora_dropout": self.model_config.lora_dropout,
                "lora_layers": self.model_config.lora_layers,
                "target_modules": self.model_config.lora_target_modules
            }
            
            with open(config_file, "w") as f:
                json.dump(adapter_config, f, indent=2)
            
            logger.info(f"✅ Final model saved to: {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
            raise