"""
Experiment tracking abstraction layer for TalkyTalky

This module provides unified interfaces for different experiment tracking backends
including Weights & Biases and TensorBoard.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentTracker(ABC):
    """Base class for experiment tracking implementations"""
    
    def __init__(self, 
                 experiment_name: str,
                 project_name: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None):
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.tags = tags or []
        self.notes = notes
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the tracking backend with configuration"""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log training metrics"""
        pass
    
    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters and configuration"""
        pass
    
    @abstractmethod
    def log_model_artifact(self, model_path: str, name: str = "model") -> None:
        """Log model artifacts"""
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """Clean up and finish tracking"""
        pass
    
    def log_text(self, text: str, key: str = "notes") -> None:
        """Log text data (optional for implementations)"""
        pass


class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker"""
    
    def __init__(self, 
                 experiment_name: str,
                 project_name: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None,
                 api_key: Optional[str] = None):
        super().__init__(experiment_name, project_name, tags, notes)
        self.api_key = api_key
        self.run = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize Weights & Biases tracking"""
        try:
            import wandb
            
            # Set API key if provided
            if self.api_key:
                wandb.login(key=self.api_key)
            
            # Initialize run
            self.run = wandb.init(
                name=self.experiment_name,
                project=self.project_name or "talkytalky-training",
                tags=self.tags,
                notes=self.notes,
                config=config,
                reinit=True
            )
            
            self.is_initialized = True
            logger.info(f"Initialized W&B tracking: {self.run.url}")
            
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize W&B tracking: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log metrics to W&B"""
        if not self.is_initialized or not self.run:
            logger.warning("W&B tracker not initialized, skipping metric logging")
            return
        
        try:
            import wandb
            self.run.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics to W&B: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to W&B"""
        if not self.is_initialized or not self.run:
            logger.warning("W&B tracker not initialized, skipping hyperparameter logging")
            return
        
        try:
            import wandb
            wandb.config.update(params, allow_val_change=True)
        except Exception as e:
            logger.error(f"Failed to log hyperparameters to W&B: {e}")
    
    def log_model_artifact(self, model_path: str, name: str = "model") -> None:
        """Log model as W&B artifact"""
        if not self.is_initialized or not self.run:
            logger.warning("W&B tracker not initialized, skipping model artifact logging")
            return
        
        try:
            import wandb
            
            # Create artifact
            artifact = wandb.Artifact(name=name, type="model")
            artifact.add_dir(model_path)
            
            # Log artifact
            self.run.log_artifact(artifact)
            logger.info(f"Logged model artifact to W&B: {name}")
            
        except Exception as e:
            logger.error(f"Failed to log model artifact to W&B: {e}")
    
    def log_text(self, text: str, key: str = "notes") -> None:
        """Log text to W&B"""
        if not self.is_initialized or not self.run:
            return
        
        try:
            import wandb
            self.run.log({key: wandb.Html(f"<p>{text}</p>")})
        except Exception as e:
            logger.error(f"Failed to log text to W&B: {e}")
    
    def finish(self) -> None:
        """Finish W&B run"""
        if self.run:
            try:
                import wandb
                self.run.finish()
                logger.info("Finished W&B tracking")
            except Exception as e:
                logger.error(f"Failed to finish W&B run: {e}")


class TensorBoardTracker(ExperimentTracker):
    """TensorBoard experiment tracker"""
    
    def __init__(self, 
                 experiment_name: str,
                 project_name: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None,
                 log_dir: Optional[str] = None):
        super().__init__(experiment_name, project_name, tags, notes)
        self.log_dir = log_dir
        self.writer = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize TensorBoard tracking"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            # Set up log directory
            if not self.log_dir:
                base_dir = "runs" if not self.project_name else f"runs/{self.project_name}"
                self.log_dir = os.path.join(base_dir, self.experiment_name)
            
            # Create writer
            self.writer = SummaryWriter(log_dir=self.log_dir)
            
            # Log hyperparameters immediately
            self.log_hyperparameters(config)
            
            self.is_initialized = True
            logger.info(f"Initialized TensorBoard tracking: {self.log_dir}")
            
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize TensorBoard tracking: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log metrics to TensorBoard"""
        if not self.is_initialized or not self.writer:
            logger.warning("TensorBoard tracker not initialized, skipping metric logging")
            return
        
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
        except Exception as e:
            logger.error(f"Failed to log metrics to TensorBoard: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard"""
        if not self.is_initialized or not self.writer:
            logger.warning("TensorBoard tracker not initialized, skipping hyperparameter logging")
            return
        
        try:
            # Convert complex types to strings for TensorBoard
            hparams = {}
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    hparams[key] = value
                else:
                    hparams[key] = str(value)
            
            # Create dummy metrics dict (required by TensorBoard)
            metrics = {"hparam/accuracy": 0}  # Will be updated during training
            
            self.writer.add_hparams(hparams, metrics)
        except Exception as e:
            logger.error(f"Failed to log hyperparameters to TensorBoard: {e}")
    
    def log_model_artifact(self, model_path: str, name: str = "model") -> None:
        """Log model path to TensorBoard (as text)"""
        if not self.is_initialized or not self.writer:
            logger.warning("TensorBoard tracker not initialized, skipping model artifact logging")
            return
        
        try:
            self.writer.add_text("model_path", f"Model saved to: {model_path}")
            logger.info(f"Logged model path to TensorBoard: {model_path}")
        except Exception as e:
            logger.error(f"Failed to log model artifact to TensorBoard: {e}")
    
    def log_text(self, text: str, key: str = "notes") -> None:
        """Log text to TensorBoard"""
        if not self.is_initialized or not self.writer:
            return
        
        try:
            self.writer.add_text(key, text)
        except Exception as e:
            logger.error(f"Failed to log text to TensorBoard: {e}")
    
    def finish(self) -> None:
        """Close TensorBoard writer"""
        if self.writer:
            try:
                self.writer.close()
                logger.info("Finished TensorBoard tracking")
            except Exception as e:
                logger.error(f"Failed to close TensorBoard writer: {e}")


class MultiTracker(ExperimentTracker):
    """Tracker that combines multiple tracking backends"""
    
    def __init__(self, trackers: List[ExperimentTracker]):
        # Use the first tracker's config for the base class
        first_tracker = trackers[0] if trackers else None
        super().__init__(
            experiment_name=first_tracker.experiment_name if first_tracker else "multi_tracker",
            project_name=first_tracker.project_name if first_tracker else None,
            tags=first_tracker.tags if first_tracker else None,
            notes=first_tracker.notes if first_tracker else None
        )
        self.trackers = trackers
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize all tracking backends"""
        for tracker in self.trackers:
            try:
                tracker.initialize(config)
            except Exception as e:
                logger.error(f"Failed to initialize tracker {type(tracker).__name__}: {e}")
        
        self.is_initialized = any(tracker.is_initialized for tracker in self.trackers)
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log metrics to all trackers"""
        for tracker in self.trackers:
            if tracker.is_initialized:
                tracker.log_metrics(metrics, step)
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to all trackers"""
        for tracker in self.trackers:
            if tracker.is_initialized:
                tracker.log_hyperparameters(params)
    
    def log_model_artifact(self, model_path: str, name: str = "model") -> None:
        """Log model artifacts to all trackers"""
        for tracker in self.trackers:
            if tracker.is_initialized:
                tracker.log_model_artifact(model_path, name)
    
    def log_text(self, text: str, key: str = "notes") -> None:
        """Log text to all trackers"""
        for tracker in self.trackers:
            if tracker.is_initialized:
                tracker.log_text(text, key)
    
    def finish(self) -> None:
        """Finish all trackers"""
        for tracker in self.trackers:
            if tracker.is_initialized:
                tracker.finish()


def create_tracker(config) -> Optional[ExperimentTracker]:
    """Factory function to create appropriate tracker based on configuration"""
    if config.tracking_provider == "none":
        return None
    
    trackers = []
    
    # Create W&B tracker if requested
    if config.tracking_provider in ["wandb", "both"]:
        wandb_tracker = WandbTracker(
            experiment_name=config.experiment_name,
            project_name=config.project_name,
            tags=config.tracking_tags,
            notes=config.tracking_notes,
            api_key=config.wandb_api_key
        )
        trackers.append(wandb_tracker)
    
    # Create TensorBoard tracker if requested
    if config.tracking_provider in ["tensorboard", "both"]:
        tb_tracker = TensorBoardTracker(
            experiment_name=config.experiment_name,
            project_name=config.project_name,
            tags=config.tracking_tags,
            notes=config.tracking_notes,
            log_dir=config.logging_dir
        )
        trackers.append(tb_tracker)
    
    # Return appropriate tracker
    if len(trackers) == 0:
        return None
    elif len(trackers) == 1:
        return trackers[0]
    else:
        return MultiTracker(trackers)