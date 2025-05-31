#!/usr/bin/env python3
import os
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

def setup_logging(script_name, debug_mode=False):
    """
    Configure logging for any script that uses it.
    
    Args:
        script_name (str): Name of the script (e.g., 'transcribe', 'diarize', 'summarize')
        debug_mode (bool): Whether to enable debug logging
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a rotating file handler
    log_file = log_dir / f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the logger for this script
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Ensure handlers are properly configured
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.flush = handler.stream.flush
    
    return logger 