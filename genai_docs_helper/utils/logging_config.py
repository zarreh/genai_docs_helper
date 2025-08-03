import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "genai_docs_helper",
    log_level: str = "INFO",
    log_dir: str = "./logs",
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
) -> logging.Logger:
    """
    Set up consistent logging configuration for the application.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        enable_file_logging: Whether to log to file
        enable_console_logging: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(simple_formatter)  # Use detailed formatter for console
        logger.addHandler(console_handler)
    
    # File handler
    if enable_file_logging:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create unique log file for each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"genai_docs_helper_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Also create a latest.log symlink for convenience
        latest_log = log_path / "latest.log"
        if latest_log.exists():
            latest_log.unlink()
        try:
            latest_log.symlink_to(log_file.name)
        except (OSError, NotImplementedError):
            # Symlinks might not work on all systems
            pass
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with consistent configuration.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    # Check if root logger is already configured
    root_logger = logging.getLogger("genai_docs_helper")
    if not root_logger.handlers:
        # Set up root logger if not already configured
        setup_logging()
    
    # Return a child logger
    return logging.getLogger(f"genai_docs_helper.{name}")


def log_performance_metrics(logger: logging.Logger, metrics: dict, request_id: Optional[str] = None):
    """
    Log performance metrics in a consistent format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of performance metrics
        request_id: Optional request identifier
    """
    request_prefix = f"[Request: {request_id}] " if request_id else ""
    
    logger.info(f"{request_prefix}Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
