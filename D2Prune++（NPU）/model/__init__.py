from model.llama import LLAMA
from model.opt import OPT
from model.qwen import Qwen
from model.mistral import Mistral
from model.deit import DeiT
from model.pangu import Pangu
from utils import timeit
import os
import torch


@timeit
def get_model(model_path: str, device_type: str, seq_len=None):
    '''
    :param model_path: -> model file path or huggingface model name
    :param device_type: 'cpu', 'cuda:0', 'npu:0', 'auto'
    :param seq_len: 2048 if save memory, else None (default)
    :return: model and tokenizer
    '''
    # Check if it's NPU device
    is_npu = False
    if device_type and 'npu' in device_type.lower():
        is_npu = True
        try:
            import torch_npu
            if torch.npu.is_available():
                print(f"[INFO] NPU detected. Using device: {device_type}")
        except ImportError:
            print("[WARNING] torch_npu not found, but NPU device specified")
    
    # Check if it's a local path (starting with ../ or / or .)
    is_local_path = model_path.startswith(('../', '/', '.')) or os.path.isabs(model_path)
    
    # Set HuggingFace mirror, but only effective for remote models
    hf_mirror = os.getenv('HF_MIRROR', None)
    original_hf_endpoint = os.environ.get('HF_ENDPOINT')
    
    if hf_mirror and not is_local_path:
        # Set environment variable to let transformers library use mirror
        os.environ['HF_ENDPOINT'] = hf_mirror
        print(f"[INFO] Using HuggingFace mirror: {hf_mirror}")
    elif hf_mirror and is_local_path:
        print(f"[INFO] Local model path detected, ignoring HF mirror: {hf_mirror}")
    
    try:
        if "llama" in model_path.lower():
            llm = LLAMA(model_path, device_type)
            model = llm.load_model(seq_len)
            tokenizer = llm.load_tokenizer()
            return model, tokenizer
        elif "opt" in model_path:
            llm = OPT(model_path, device_type)
            model = llm.load_model(seq_len)
            tokenizer = llm.load_tokenizer()
            return model, tokenizer
        elif "qwen" in model_path.lower():
            llm = Qwen(model_path, device_type)
            model = llm.load_model(seq_len=seq_len if seq_len else 4096)
            tokenizer = llm.load_tokenizer()
            return model, tokenizer
        elif "mistral" in model_path.lower():
            llm = Mistral(model_path, device_type)
            model = llm.load_model(seq_len=seq_len if seq_len else 4096)
            tokenizer = llm.load_tokenizer()
            return model, tokenizer
        elif "pangu" in model_path.lower():
            llm = Pangu(model_path, device_type)
            model = llm.load_model(seq_len=seq_len if seq_len else 4096)
            tokenizer = llm.load_tokenizer()
            return model, tokenizer
    finally:
        # Restore original HF_ENDPOINT setting
        if original_hf_endpoint is not None:
            os.environ['HF_ENDPOINT'] = original_hf_endpoint
        elif 'HF_ENDPOINT' in os.environ and not is_local_path:
            del os.environ['HF_ENDPOINT']
    
    raise ValueError(f'Unknown model {model_path}')