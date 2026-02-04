"""
Pangu-Embedded-7B
Large language model based on Ascend NPU
"""
from typing import List
import os

import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from .base_model import LLM

# Pangu model type
try:
    from modeling_openpangu_dense import PanguEmbeddedForCausalLM
    PANGU_MODEL = PanguEmbeddedForCausalLM
except ImportError:
    # If direct import fails, use AutoModel
    PANGU_MODEL = None


class Pangu(LLM):
    """
    Initialize Pangu model and tokenizer.
    Args:
        model_path (str): Path to the model, such as '../cache/llm_weights/pangu'
        device_type (str): Device to load the model on. ['cpu', 'cuda:0', 'npu:0', 'auto']
        model_name (str): Optional model name
    """
    def __init__(self, model_path, device_type, model_name=None):
        super().__init__(model_path)
        self.device_type = device_type
        self.model_name = model_name.lower() if model_name else model_path.split("/")[-1]
        
        # Check NPU support
        self.is_npu = False
        if device_type and ('npu' in device_type.lower() or device_type == 'auto'):
            try:
                import torch_npu
                if torch.npu.is_available():
                    self.is_npu = True
                    print(f"[INFO] NPU detected and available. Device: {device_type}")
            except ImportError:
                print("[WARNING] torch_npu not found, falling back to CPU/CUDA")

    def load_model(self, seq_len=None):
        """
        Load Pangu model
        Support NPU devices
        """
        # Convert relative path to absolute path
        if not os.path.isabs(self.model_path):
            # If it's a relative path, convert to absolute path
            model_path = os.path.abspath(self.model_path)
        else:
            model_path = self.model_path
        
        # Check if path exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Check if it's a directory (local model)
        if not os.path.isdir(model_path):
            raise ValueError(f"Model path is not a directory: {model_path}")
        
        print(f"[INFO] Loading model from: {model_path}")
        
        # Handle NPU device type
        if self.is_npu and self.device_type == 'auto':
            # For NPU, using device_map='auto' may not work, need manual specification
            device_map = 'auto'
        elif self.is_npu and 'npu' in self.device_type:
            # Directly use npu device
            device_map = self.device_type
        else:
            device_map = self.device_type
            
        if seq_len:  # Save memory
            def skip(*args, **kwargs):
                pass
            torch.nn.init.kaiming_uniform_ = skip
            torch.nn.init.uniform_ = skip
            torch.nn.init.normal_ = skip
            
            # Try to load from local path (including custom model files)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,  # Using absolute path
                    torch_dtype='auto',
                    low_cpu_mem_usage=True,
                    device_map=device_map,
                    trust_remote_code=True,  # Pangu requires custom code
                    local_files_only=True     # Force using local files
                )
            except Exception as e:
                print(f"[WARNING] Failed to load with device_map, trying CPU first: {e}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,  # Using absolute path
                    torch_dtype='auto',
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    local_files_only=True      # Force using local files
                )
                if self.is_npu and 'npu' in str(self.device_type):
                    # Manually move to NPU
                    device_id = int(self.device_type.split(':')[-1]) if ':' in self.device_type else 0
                    model = model.to(f'npu:{device_id}')
            
            model.seq_len = seq_len
            return model

        # Normal loading
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,  # Using absolute path
                cache_dir=self.cache_dir,
                torch_dtype=torch.bfloat16 if self.is_npu else torch.float16,
                low_cpu_mem_usage=True,
                device_map=device_map,
                trust_remote_code=True,  # Pangu requires custom code
                local_files_only=True    # Force using local files
            )
        except Exception as e:
            print(f"[WARNING] Failed to load with device_map, trying CPU first: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,  # Using absolute path
                cache_dir=self.cache_dir,
                torch_dtype=torch.bfloat16 if self.is_npu else torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                local_files_only=True     # Force using local files
            )
            if self.is_npu and 'npu' in str(self.device_type):
                # Manually move to NPU
                device_id = int(self.device_type.split(':')[-1]) if ':' in self.device_type else 0
                model = model.to(f'npu:{device_id}')
        
        # Set sequence length
        model.seq_len = 4096  # Set Pangu maximum sequence length
        
        return model

    def load_tokenizer(self):
        """Load Pangu tokenizer"""
        # Convert relative path to absolute path
        if not os.path.isabs(self.model_path):
            model_path = os.path.abspath(self.model_path)
        else:
            model_path = self.model_path
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,  # Using absolute path
            cache_dir=self.cache_dir,
            use_fast=False,
            legacy=False,
            trust_remote_code=True,  # Pangu requires custom code
            local_files_only=True     # Force using local files
        )
        return tokenizer

    def load_layers(self, model, model_type) -> List[torch.nn.Module]:
        """Load model layers"""
        # Pangu model structure is similar to LLaMA, using model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return [layer for layer in model.model.layers]
        else:
            raise ValueError(f'Unknown Pangu model structure. Expected model.model.layers')

    def load_embedding(self, model, model_type) -> List[torch.nn.Module]:
        """Load embedding layer"""
        # Pangu model structure is similar to LLaMA
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            return [model.model.embed_tokens]
        else:
            raise ValueError(f'Unknown Pangu model structure. Expected model.model.embed_tokens')

