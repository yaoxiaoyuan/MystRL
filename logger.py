#encoding=utf-8
#◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Author: Xiaoyuan Yao
# GitHub: https://github.com/yaoxiaoyuan/mystRL/
# Contact: yaoxiaoyuan1990@gmail.com
# Created: Sat Jun 14 15:08:00 2025
# License: MIT
# Version: 0.1.0
#
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
import os
import math
import logging
import random
import time
import json
import argparse
import numpy as np

def build_logger():
    """
    Initialize and configure a logger with console output handling.

    Returns:
        logging.Logger: Configured logger instance with console handler
    """
    logger = logging.getLogger(os.path.basename(__file__))
    logger.setLevel(logging.DEBUG)

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
 
    logger.addHandler(console_handler)                                                               
    return logger

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logger")
if not os.path.exists(LOG_DIR):
    try:
        os.mkdir(LOG_DIR)
    except:
        pass

logger = build_logger() 

def add_file_handlers(log_path):                                                                     
    """                                                                                              
    Add file handlers to global logger for writing DEBUG+ logs to specified path                     
                                                                                                     
    Parameters:                                                                                      
    log_path (str) : Target file path for logs (use absolute path recommended)                       
    """                                                                                              
    file_handler = logging.FileHandler(log_path)                                                     
    file_handler.setLevel(logging.DEBUG)                                                             
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')          
    file_handler.setFormatter(file_format)                                                           
    logger.addHandler(file_handler)  


def print_formated_args(args):
    """
    Prints configuration arguments in human-readable JSON format.

    Formats arguments for debugging/configuration tracking purposes. Ensures
    consistent output in distributed environments through rank-aware logging.

    Args:
        args (Namespace): Configuration arguments object (typically from argparse)
    """
    formatted_args = json.dumps(vars(args), ensure_ascii=False, indent=4)
    logger.info(f"List all args: {formatted_args}")
   

