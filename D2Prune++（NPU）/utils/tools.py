import os
import platform
import random
import sys
import time
from datetime import datetime
from functools import wraps

import numpy as np
import toml
import torch
import torch.nn as nn
import yaml
from loguru import logger


## yaml
def read_yaml_to_dict(yaml_path: str):
    """yaml file to dict"""
    with open(yaml_path, 'r') as file:
        dict_value = yaml.safe_load(file)
        return dict_value

def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict save to yaml"""
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))

## toml
def read_toml_to_dict(toml_path: str):
    """toml file to dict"""
    with open(toml_path, 'r') as file:
        dict_value = toml.load(file)
        return dict_value

def save_toml_to_dict(dict_value: dict, save_path: str):
    """dict save to toml"""
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(toml.dumps(dict_value))


def setup_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup_logger(log_name, save_dir, args_model=None, args_device=None):
    filename = '%s.log' % log_name
    save_file = os.path.join(save_dir, filename)
    if os.path.exists(save_file):
        with open(save_file, "w") as log_file:
            log_file.truncate()
    logger.remove()
    logger.add(save_file, rotation="10 MB", format="{time} {level} {message}", level="INFO")
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.info('This is the %s log' % log_name)
    
    # Add hardware information to log
    try:
        model_name = args_model.split("/")[-1] if args_model else "Unknown"
        server_model = get_server_model()
        device_info = get_device_info(args_device)
        logger.info(f'Model Name: {model_name}')
        logger.info(f'Server Model: {server_model}')
        logger.info(f'Device Info: {device_info}')
    except Exception as e:
        # If getting system information fails, at least output model name
        model_name = args_model.split("/")[-1] if args_model else "Unknown"
        logger.warning(f"Failed to get system info: {e}")
        logger.info(f'Model Name: {model_name}')
        logger.info(f'Server Model: Unknown')
        logger.info(f'Device Info: Unknown')
    
    return logger


def timeit(func):
    """
    running time evaluation
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Function '{func.__name__}' took {duration:.6f} seconds to run.")
        return result
    return wrapper


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.decoder.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            # count += (W==0).sum().item()
            count += (W == 0).sum().cpu().item()
            total_params += W.numel()
            # sub_count += (W == 0).sum().item()
            sub_count += (W==0).sum().cpu().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count)/total_params


def get_server_model():
    """
    Get server model
    Supports Linux and Windows systems, especially Huawei servers
    """
    server_model = "Unknown"
    
    # Try to read from DMI information (Linux system)
    dmi_paths = [
        '/sys/class/dmi/id/product_name',
        '/sys/class/dmi/id/board_name',
        '/sys/class/dmi/id/product_version'
    ]
    
    for path in dmi_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    server_model = f.read().strip()
                    if server_model and server_model != "Unknown" and server_model:
                        # Clean up possible whitespace characters
                        server_model = server_model.strip()
                        break
            except Exception:
                continue
    
    # Huawei server specific paths (Ascend server)
    if server_model == "Unknown" and platform.system() == "Linux":
        huawei_paths = [
            '/sys/class/dmi/id/product_name',  # Priority check DMI (may contain Huawei server information)
            '/proc/device-tree/model',  # Device tree model information
            '/sys/firmware/devicetree/base/model',  # Device tree base model
            '/etc/hw_info',  # Huawei server hardware information file
        ]
        for path in huawei_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                        if content and content:
                            # For /etc/hw_info, may need to parse specific fields
                            if path == '/etc/hw_info':
                                # Try to find lines containing "model" or "server"
                                for line in content.split('\n'):
                                    if 'model' in line.lower() or 'server' in line.lower():
                                        parts = line.split('=')
                                        if len(parts) > 1:
                                            server_model = parts[1].strip()
                                            break
                                if server_model == "Unknown":
                                    # If not found, use first non-comment line
                                    for line in content.split('\n'):
                                        if line.strip() and not line.strip().startswith('#'):
                                            server_model = line.strip()
                                            break
                            else:
                                server_model = content.strip()
                            if server_model and server_model != "Unknown":
                                break
                except Exception:
                    continue
    
    # If Linux method fails, try using platform module
    if server_model == "Unknown":
        try:
            # Windows system
            if platform.system() == "Windows":
                import subprocess
                try:
                    result = subprocess.run(
                        ['wmic', 'computersystem', 'get', 'model'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            server_model = lines[1].strip()
                            if server_model:
                                return server_model
                except Exception:
                    pass
        except Exception:
            pass
    
    # If still unknown, use platform information
    if server_model == "Unknown" or not server_model:
        server_model = f"{platform.system()} {platform.machine()}"
    
    return server_model


def get_device_info(device_str=None):
    """
    Get GPU or NPU device information
    Args:
        device_str: Device string, such as 'cuda:0', 'npu:0', etc. (optional, mainly for logging)
    Returns:
        str: Device information string, format like "Ascend910B2 (x1)" or "NVIDIA A100 (x2)"
    """
    device_info = "Unknown Device"
    try:
        # Check for NPU (Ascend) - Priority check NPU
        if hasattr(torch, "npu") and torch.npu.is_available():
            # Get NPU device count and model
            npu_count = torch.npu.device_count()
            if npu_count > 0:
                # Get first NPU device information
                try:
                    device_name = torch.npu.get_device_name(0) if hasattr(torch.npu, 'get_device_name') else "Ascend NPU"
                    device_info = f"{device_name} (x{npu_count})"
                except Exception:
                    device_info = f"Ascend NPU (x{npu_count})"
        # Check for CUDA GPU
        elif torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                try:
                    device_name = torch.cuda.get_device_name(0)
                    device_info = f"{device_name} (x{gpu_count})"
                except Exception:
                    device_info = f"CUDA Device (x{gpu_count})"
    except Exception:
        device_info = "Unknown Device"
    
    return device_info