"""
Logging utilities for the LLM benchmark system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the entire application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    
    # Add file handler if specified
    handlers = [console_handler]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger with handlers
    root_logger = logging.getLogger()
    root_logger.handlers = handlers


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class BenchmarkLogger:
    """
    Specialized logger for benchmark operations with progress tracking.
    """
    
    def __init__(self, name: str, total_tasks: Optional[int] = None):
        self.logger = get_logger(name)
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        
    def log_progress(self, message: str, level: str = "INFO") -> None:
        """Log progress with task completion info"""
        if self.total_tasks:
            progress = (self.completed_tasks / self.total_tasks) * 100
            full_message = f"[{progress:.1f}%] {message}"
        else:
            full_message = message
            
        getattr(self.logger, level.lower())(full_message)
    
    def log_task_start(self, task_name: str) -> None:
        """Log the start of a task"""
        self.log_progress(f"Starting: {task_name}")
    
    def log_task_complete(self, task_name: str, success: bool = True) -> None:
        """Log the completion of a task"""
        self.completed_tasks += 1
        status = "✓" if success else "✗"
        self.log_progress(f"Completed: {task_name} {status}")
    
    def log_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log an error with optional exception details"""
        if exception:
            self.logger.error(f"{message}: {str(exception)}")
        else:
            self.logger.error(message)
    
    def log_summary(self, passed: int, total: int) -> None:
        """Log a summary of results"""
        success_rate = (passed / total) * 100 if total > 0 else 0
        self.logger.info(f"Summary: {passed}/{total} passed ({success_rate:.1f}%)")


class PerformanceLogger:
    """
    Logger for performance metrics and timing information.
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.timings = {}
        
    def start_timing(self, operation: str) -> None:
        """Start timing an operation"""
        import time
        self.timings[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timing(self, operation: str) -> float:
        """End timing an operation and return duration"""
        import time
        if operation not in self.timings:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.timings[operation]
        del self.timings[operation]
        
        self.logger.info(f"Operation '{operation}' took {duration:.2f}s")
        return duration
    
    def log_memory_usage(self, operation: str) -> None:
        """Log current memory usage"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Memory usage after {operation}: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.debug("psutil not available, skipping memory logging")
    
    def log_system_info(self) -> None:
        """Log system information"""
        try:
            import psutil
            import platform
            
            self.logger.info(f"Python version: {platform.python_version()}")
            self.logger.info(f"Platform: {platform.platform()}")
            self.logger.info(f"CPU cores: {psutil.cpu_count()}")
            self.logger.info(f"Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
        except ImportError:
            self.logger.debug("psutil not available, skipping system info logging")


def configure_benchmark_logging(
    results_dir: Path,
    log_level: str = "INFO",
    enable_file_logging: bool = True
) -> None:
    """
    Configure logging specifically for benchmark runs.
    
    Args:
        results_dir: Directory to store log files
        log_level: Logging level
        enable_file_logging: Whether to enable file logging
    """
    log_file = None
    if enable_file_logging:
        log_file = results_dir / "benchmark.log"
    
    setup_logging(
        level=log_level,
        log_file=log_file,
        format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Also log to a separate file for errors only
    if enable_file_logging:
        error_log_file = results_dir / "benchmark_errors.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(error_formatter)
        
        # Add error handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(error_handler)