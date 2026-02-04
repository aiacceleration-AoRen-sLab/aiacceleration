import torch.nn as nn
import torch
import gc
import time
import os
from utils import timeit
from tqdm import tqdm, trange
import numpy as np

from .d2prune_utils import D2SparseGPT, D2Wanda, D2ADMM
from .pruner_zero import PrunerZero
from .sparsegpt import SparseGPT
from .wanda import Wanda
from .admm_grad import AdmmGrad

class D2Prune_PANGU:
    '''
    D2Prune for Pangu model:
    1. using 1st-order activation derivatives and 2nd-order weights derivatives for pruning metric
    2. attention awareness: q/k/v weights hybrid update (D2SparseGPT) or no-update (D2Wanda)
    '''

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.device = args.device  # 'cpu', 'cuda:0', 'npu:0'
        self.sparsity_ratio = args.sparsity_ratio
        self.nsamples = args.nsamples
        self.target_layer_names = args.target_layer_names  # []
        self.d2_sparsegpt = args.d2_sparsegpt
        self.d2_wanda = args.d2_wanda
        self.d2_admm = args.d2_admm
        self.prune_n = args.prune_n
        self.prune_m = args.prune_m
        self.logger = self.args.logger
        
        # Check if it's NPU device
        self.is_npu = False
        if 'npu' in str(self.device).lower():
            self.is_npu = True
            try:
                import torch_npu
                if torch.npu.is_available():
                    self.logger.info(f"Using NPU device: {self.device}")
            except ImportError:
                self.logger.warning("torch_npu not found, but NPU device specified")

    def init_model(self):
        self.model.eval()
        self.use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        # Pangu model structure is similar to LLaMA
        self.layers = self.model.model.layers

    @classmethod
    def find_layers(cls, module, layers=[nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(cls.find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res

    def check_sparsity(self, tolerance=1e-6):
        self.model.config.use_cache = False
        count = 0
        total_params = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            subset = self.find_layers(layer)
            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W == 0).sum().cpu().item()
                total_params += W.numel()
                sub_count += (W == 0).sum().cpu().item()
                sub_params += W.numel()
            self.logger.info(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
        self.model.config.use_cache = self.use_cache
        error = abs(float(count) / total_params - self.sparsity_ratio)
        if error <= tolerance:
            self.logger.info("Pruning correctly executed")
        else:
            self.logger.info("Pruning not performed correctly")
        return float(count)/total_params

    @staticmethod
    def check_outlier_mean(mask, threshold):
        W = mask
        count = 0
        total_params = 0
        max_shred = torch.mean(W) * threshold
        count += (W > max_shred).sum().item()
        total_params += W.numel()
        outlier_ratio = float(count) / total_params * 100
        return outlier_ratio
    
    @torch.no_grad()
    def get_layer_dynamic_sparsity(self, subset, gpts_layers, wrapped_layers, dsm='owl', granularity='per-block'):
        """
        Sparsity compensation
        Compensate for over-pruning caused by uniform sparsity due to different layer sensitivities, and balance sparsity.
        :param dsm:dynamic sparsity method-->global static adjustments
        :return:subset each layer sparsity
        """
        if dsm == "owl":
            if granularity == 'per-block':
                self.layer_outlier_ratios = []
                self.block_sizes = []
                for name in subset:
                    if name in self.target_layer_names:
                        gpts = wrapped_layers
                    else:
                        gpts = gpts_layers
                    W_metric = (torch.abs(gpts[name].layer.weight.data) ** 2) * (
                                gpts[name].scaler_row.reshape((1, -1)) ** (1))
                    if self.args.d2_wanda:
                        W_metric += (gpts[name].r1) * (gpts[name].y_scaler_col.reshape((-1, 1)) ** (1)) * (
                            torch.abs(gpts[name].layer.weight.data)) * (
                                                gpts[name].delta_x_scaler_row.reshape((1, -1)) ** (0))
                        W_metric += -(gpts[name].r2) * (torch.abs(gpts[name].layer.weight.data) ** (2)) * (
                                    gpts[name].delta_x_scaler_row.reshape((1, -1)) ** (2))
                    
                    block_outlier_ratio = self.check_outlier_mean(torch.flatten(W_metric.cpu()),
                                                                  self.args.Hyper_m)
                    self.layer_outlier_ratios.append(block_outlier_ratio)
                    self.block_sizes.append(subset[name].weight.numel())  
                total_params = sum(self.block_sizes)
                block_weights = np.array(self.block_sizes) / total_params
                self.all_blocks_ratio = np.array(self.layer_outlier_ratios)
                self.all_blocks_ratio = (self.all_blocks_ratio - self.all_blocks_ratio.min()) / (self.all_blocks_ratio.max() - self.all_blocks_ratio.min())
                target_sparsity = self.args.sparsity_ratio
                delta = (self.all_blocks_ratio - np.mean(self.all_blocks_ratio)) * self.args.Lambda * 2
                self.all_blocks_ratio = np.clip(target_sparsity + delta, 0.1, 0.95)  

                current_weighted_sparsity = np.sum(self.all_blocks_ratio * block_weights)
                scale = target_sparsity / current_weighted_sparsity
                self.all_blocks_ratio = 1-np.clip(self.all_blocks_ratio * scale, 0.1, 0.95)

                self.logger.info(f"Block sparsity: {1-self.all_blocks_ratio}, "
                                 f"Block outlier ratio: {self.all_blocks_ratio}, "
                                 f"Target sparsity: {target_sparsity:.4f}, "
                                 f"Weighted sparsity: {np.sum((1-self.all_blocks_ratio) * block_weights):.4f}, ")
                self.logger.info("before layer sparsity compensation", self.layer_outlier_ratios)
                return self.all_blocks_ratio
            elif granularity == 'per-layer':
                self.layer_wmetric = []
                for name in subset:
                    if name in self.target_layer_names:
                        gpts = wrapped_layers
                    else:
                        gpts = gpts_layers
                    W_metric = (torch.abs(gpts[name].layer.weight.data) ** 2) * (
                            gpts[name].scaler_row.reshape((1, -1)) ** (1))
                    if self.args.d2_wanda:
                        W_metric += (gpts[name].r1) * (gpts[name].y_scaler_col.reshape((-1, 1)) ** (1)) * (
                            torch.abs(gpts[name].layer.weight.data)) * (
                                            gpts[name].delta_x_scaler_row.reshape((1, -1)) ** (0))
                        W_metric += -(gpts[name].r2) * (torch.abs(gpts[name].layer.weight.data) ** (2)) * (
                                gpts[name].delta_x_scaler_row.reshape((1, -1)) ** (2))
                    self.layer_wmetric.append(torch.flatten(W_metric.cpu()))
                self.layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in self.layer_wmetric])
                self.out_ratio_layer = self.check_outlier_mean(self.layer_wmetric, self.args.Hyper_m)
                return self.out_ratio_layer

    @torch.no_grad()
    def prepare_layer_calibration(self, train_loader, layer_ind=0):
        '''
        use device == embed_tokens.weight.device, if cpu, turn to specified device
        '''
        device = self.model.model.embed_tokens.weight.device  
        if device.type == 'cpu':
            # Parse device string
            if ':' in str(self.device):
                device_str = str(self.device)
            elif self.is_npu:
                device_id = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0')
                device_str = f'npu:{device_id}'
            else:
                device_str = 'cuda:0'
            device = torch.device(device_str)
            self.model.model.embed_tokens.to(device)
        else:
            device = device
        
        self.logger.info(f"using device to calibrate-->device: {device}")

        dtype = next(iter(self.model.parameters())).dtype  # torch.bfloat16 or torch.float16
        inps = torch.zeros((self.nsamples, self.model.seq_len, self.model.config.hidden_size), dtype=dtype,
                                device=device)
        inps.requires_grad = False
        cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs.get('attention_mask', None)
                cache['position_ids'] = kwargs.get('position_ids', None)
                # Pangu model requires position_embeddings (RoPE's cos/sin tensors)
                cache['position_embeddings'] = kwargs.get('position_embeddings', None)
                raise ValueError
        
        self.layers[layer_ind] = Catcher(self.layers[layer_ind])
        for batch in train_loader:  
            try:
                self.model(batch[0].reshape(-1, self.model.seq_len).to(device))  # batch[0]-->[1,2048]
            except ValueError:
                pass
        self.layers[layer_ind] = self.layers[layer_ind].module
        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        position_embeddings = cache['position_embeddings']
        
        # If position_embeddings is None, try to get from model's rope module
        if position_embeddings is None and hasattr(self.model.model, 'layers') and len(self.model.model.layers) > 0:
            first_layer = self.model.model.layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'rotary_emb'):
                rope = first_layer.self_attn.rotary_emb
                if position_ids is not None:
                    try:
                        cos, sin = rope(position_ids)
                        position_embeddings = (cos, sin)
                        self.logger.info("Computed position_embeddings from RoPE module in calibration")
                    except Exception as e:
                        self.logger.warning(f"Failed to compute position_embeddings from RoPE: {e}")
        
        self.model.config.use_cache = self.use_cache  # True
        if self.args.free:
            self.model.model.embed_tokens.to("cpu")
        
        # Clear cache (adapted for NPU and CUDA)
        if self.is_npu:
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()
        
        return inps, outs, attention_mask, position_ids, position_embeddings

    def forward_layer_wrapper(self, layer, inps, outs, attention_mask, position_ids, position_embeddings=None):
        subset = self.find_layers(layer)
        gpts = {}
        wrapped_layers = {}
        for name in subset:
            if name not in self.target_layer_names:
                if self.args.d2_sparsegpt:
                    gpts[name] = D2SparseGPT(self.args, subset[name])
                elif self.args.d2_admm:
                    gpts[name] = D2ADMM(self.args, subset[name])
                else:
                    gpts[name] = SparseGPT(self.args, subset[name])
            else:
                if self.args.d2_wanda:
                    wrapped_layers[name] = D2Wanda(self.args, subset[name])
                else:
                    wrapped_layers[name] = Wanda(self.args, subset[name])

        def add_batch_sparsegpt(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        def add_batch_wrapped_gpt(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles_sparsegpt = []
        handles_wrapped_gpt = []
        for name in subset:
            if name not in self.target_layer_names:
                handles_sparsegpt.append(subset[name].register_forward_hook(add_batch_sparsegpt(name)))
            else:
                handles_wrapped_gpt.append(subset[name].register_forward_hook(add_batch_wrapped_gpt(name)))
        
        # If position_embeddings is None, try to get from layer's rope module
        if position_embeddings is None and hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
            rope = layer.self_attn.rotary_emb
            if position_ids is not None:
                try:
                    cos, sin = rope(position_ids)
                    position_embeddings = (cos, sin)
                except Exception as e:
                    self.logger.warning(f"Failed to compute position_embeddings from layer RoPE: {e}")
        
        for j in range(inps.shape[0]):
            with torch.no_grad():  # [1,2048,4096]
                # Pangu model requires passing position_embeddings parameter (RoPE's cos/sin tensors)
                if position_embeddings is not None:
                    # Ensure position_embeddings is on the correct device
                    if isinstance(position_embeddings, tuple):
                        cos, sin = position_embeddings
                        cos = cos.to(inps.device) if hasattr(cos, 'to') else cos
                        sin = sin.to(inps.device) if hasattr(sin, 'to') else sin
                        current_position_embeddings = (cos, sin)
                    else:
                        current_position_embeddings = position_embeddings
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                   position_ids=position_ids, position_embeddings=current_position_embeddings)[0]
                else:
                    # If still None, try not passing (some versions may not need it)
                    try:
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                    except TypeError as e:
                        self.logger.error(f"Layer requires position_embeddings but it's None: {e}")
                        raise ValueError("Pangu model requires position_embeddings (RoPE cos/sin tensors). "
                                       "Please ensure the model forward pass provides them.") from e
        
        for h in handles_sparsegpt:
            h.remove()
        for h in handles_wrapped_gpt:
            h.remove()
        return subset, gpts, wrapped_layers

    @timeit
    def prune_layer_weight(self, subset, gpts, wrapped_layers):
        for i, name in enumerate(subset):
            if self.args.dsm != None:
                if self.args.granularity == 'per-block':
                    self.sparsity_ratio = 1 - self.all_layers_blocks_ratio[int(self.index_layer.split('_')[-1])][i]
                    self.logger.info(f"block sparsity  compensate, origin sparsity:{self.args.sparsity_ratio}->new sparsity:{self.sparsity_ratio}")
                elif self.args.granularity == 'per-layer':
                    self.sparsity_ratio = 1-self.all_layers_ratio[int(self.index_layer.split('_')[-1])]
                    self.logger.info(f"layer sparsity  compensate, origin sparsity:{self.args.sparsity_ratio}->new sparsity:{self.sparsity_ratio}")
            if name not in self.target_layer_names: # update weights
                if self.d2_sparsegpt:
                    self.logger.info(f"pruning {name} by D2-SparseGPT: r1={self.args.r1}, r2={self.args.r2}")
                elif self.d2_admm:
                    self.logger.info(f"pruning {name} by D2_Admm")
                else:
                    self.logger.info(f"pruning {name} by SparseGPT")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m)
                gpts[name].free()
            else:
                if self.d2_wanda:
                    self.logger.info(f"pruning {name} by D2-Wanda: r1={self.args.r1}, r2={self.args.r2}")
                else:
                    self.logger.info(f"pruning {name} by Wanda")
                wrapped_layers[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m)
                wrapped_layers[name].free()
            
            # Clear cache (adapted for NPU and CUDA)
            if self.is_npu:
                torch.npu.empty_cache()
            else:
                torch.cuda.empty_cache()

    @timeit
    def prune_llm(self, train_loader):
        self.init_model()
        inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)
        self.all_layers_ratio = []
        self.all_layers_blocks_ratio = []
        for i in tqdm(range(len(self.layers)), desc='Pruning Processing'):
            layer = self.layers[i]
            self.index_layer = f'layer_{i}'
            
            # Handle device mapping (supports multiple devices)
            if hasattr(self.model, 'hf_device_map') and f"model.layers.{i}" in self.model.hf_device_map:
                dev = self.model.hf_device_map[f"model.layers.{i}"]
                if attention_mask is not None:
                    inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                else:
                    inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)
                # Handle position_embeddings device
                if position_embeddings is not None and isinstance(position_embeddings, tuple):
                    cos, sin = position_embeddings
                    cos = cos.to(dev) if hasattr(cos, 'to') else cos
                    sin = sin.to(dev) if hasattr(sin, 'to') else sin
                    position_embeddings = (cos, sin)
            elif hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj') and layer.self_attn.q_proj.weight.device.type == 'cpu':
                # Single device operation, through offload
                if ':' in str(self.device):
                    dev = torch.device(self.device)
                elif self.is_npu:
                    device_id = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0')
                    dev = torch.device(f'npu:{device_id}')
                else:
                    dev = torch.device('cuda:0')
                layer.to(dev)
                if attention_mask is not None:
                    inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                else:
                    inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)
                # Handle position_embeddings device
                if position_embeddings is not None and isinstance(position_embeddings, tuple):
                    cos, sin = position_embeddings
                    cos = cos.to(dev) if hasattr(cos, 'to') else cos
                    sin = sin.to(dev) if hasattr(sin, 'to') else sin
                    position_embeddings = (cos, sin)
            else:
                # Get the device where the layer is located
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj'):
                    dev = layer.self_attn.q_proj.weight.device
                else:
                    # Default to using specified device
                    if ':' in str(self.device):
                        dev = torch.device(self.device)
                    elif self.is_npu:
                        device_id = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0')
                        dev = torch.device(f'npu:{device_id}')
                    else:
                        dev = torch.device('cuda:0')

            start = time.time()
            # 1. forward layer wrapper
            subset, gpts, wrapped_layers = self.forward_layer_wrapper(layer, inps, outs, attention_mask,
                                                                      position_ids, position_embeddings)
            # whether to prune by dynamic sparsity-->get subset layer sparsity-->
            if self.args.dsm != None:
                if self.args.granularity == 'per-block':
                    self.all_blocks_ratio = self.get_layer_dynamic_sparsity(subset, gpts, wrapped_layers, self.args.dsm, self.args.granularity)
                    self.logger.info(f'layer {i} blocks outlier ratio{self.all_blocks_ratio}')
                    self.all_layers_blocks_ratio.append(self.all_blocks_ratio)
                elif self.args.granularity == 'per-layer':
                    self.out_ratio_layer = self.get_layer_dynamic_sparsity(subset, gpts, wrapped_layers, self.args.dsm,
                                                                           self.args.granularity)
                    self.all_layers_ratio.append(self.out_ratio_layer)
                for j in range(self.nsamples):
                    with torch.no_grad():
                        # Pass position_embeddings
                        if position_embeddings is not None:
                            if isinstance(position_embeddings, tuple):
                                cos, sin = position_embeddings
                                cos = cos.to(dev) if hasattr(cos, 'to') else cos
                                sin = sin.to(dev) if hasattr(sin, 'to') else sin
                                current_pe = (cos, sin)
                            else:
                                current_pe = position_embeddings
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                          position_ids=position_ids, position_embeddings=current_pe)[0]
                        else:
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                inps, outs = outs, inps
                del layer, subset, gpts, wrapped_layers
                gc.collect()
                if self.is_npu:
                    torch.npu.empty_cache()
                else:
                    torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    if self.is_npu:
                        torch.npu.empty_cache()
                    else:
                        torch.cuda.empty_cache()
                
            else: # uniform sparsity
                # 2. pruning weight
                self.prune_layer_weight(subset, gpts, wrapped_layers)
                for j in range(self.nsamples):
                    with torch.no_grad():
                        # Pass position_embeddings
                        if position_embeddings is not None:
                            if isinstance(position_embeddings, tuple):
                                cos, sin = position_embeddings
                                cos = cos.to(dev) if hasattr(cos, 'to') else cos
                                sin = sin.to(dev) if hasattr(sin, 'to') else sin
                                current_pe = (cos, sin)
                            else:
                                current_pe = position_embeddings
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                          position_ids=position_ids, position_embeddings=current_pe)[0]
                        else:
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                # update next layer inputs
                self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
                inps, outs = outs, inps
                del layer, subset, gpts
                gc.collect()
                if self.is_npu:
                    torch.npu.empty_cache()
                else:
                    torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    if self.is_npu:
                        torch.npu.empty_cache()
                    else:
                        torch.cuda.empty_cache()
        
        if self.all_layers_ratio or self.all_layers_blocks_ratio:
            inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)
            if self.args.granularity == 'per-layer':
                self.logger.info(self.all_layers_ratio)
                self.all_layers_ratio = np.array(self.all_layers_ratio)
                self.all_layers_ratio = ((self.all_layers_ratio - self.all_layers_ratio.min()) * (
                        1 / (self.all_layers_ratio.max() - self.all_layers_ratio.min()) * self.args.Lambda * 2))
                self.all_layers_ratio = self.all_layers_ratio - np.mean(self.all_layers_ratio) + (
                            1 - self.args.sparsity_ratio)
            for i in range(len(self.layers)):
                layer = self.layers[i]
                self.index_layer = f'layer_{i}'
                if hasattr(self.model, 'hf_device_map') and f"model.layers.{i}" in self.model.hf_device_map:
                    dev = self.model.hf_device_map[f"model.layers.{i}"]
                    inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)
                    if position_embeddings is not None and isinstance(position_embeddings, tuple):
                        cos, sin = position_embeddings
                        cos = cos.to(dev) if hasattr(cos, 'to') else cos
                        sin = sin.to(dev) if hasattr(sin, 'to') else sin
                        position_embeddings = (cos, sin)
                elif hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj') and layer.self_attn.q_proj.weight.device.type == 'cpu':
                    if ':' in str(self.device):
                        dev = torch.device(self.device)
                    elif self.is_npu:
                        device_id = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0')
                        dev = torch.device(f'npu:{device_id}')
                    else:
                        dev = torch.device('cuda:0')
                    layer.to(dev)
                    if position_embeddings is not None and isinstance(position_embeddings, tuple):
                        cos, sin = position_embeddings
                        cos = cos.to(dev) if hasattr(cos, 'to') else cos
                        sin = sin.to(dev) if hasattr(sin, 'to') else sin
                        position_embeddings = (cos, sin)
                if attention_mask is not None:
                    inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                else:
                    inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)
                start = time.time()
                # 1. forward layer wrapper
                subset, gpts, wrapped_layers = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids, position_embeddings)

                # 2. pruning weight
                self.prune_layer_weight(subset, gpts, wrapped_layers)

                # 3. forward layers
                for j in range(self.nsamples):
                    with torch.no_grad():
                        # Pass position_embeddings
                        if position_embeddings is not None:
                            if isinstance(position_embeddings, tuple):
                                cos, sin = position_embeddings
                                cos = cos.to(dev) if hasattr(cos, 'to') else cos
                                sin = sin.to(dev) if hasattr(sin, 'to') else sin
                                current_pe = (cos, sin)
                            else:
                                current_pe = position_embeddings
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                          position_ids=position_ids, position_embeddings=current_pe)[0]
                        else:
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                # update next layer inputs
                self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
                inps, outs = outs, inps

                del layer, subset, gpts, wrapped_layers
                gc.collect()
                if self.is_npu:
                    torch.npu.empty_cache()
                else:
                    torch.cuda.empty_cache()
                if self.args.free:
                    self.model.model.layers[i].to("cpu")
                    if self.is_npu:
                        torch.npu.empty_cache()
                    else:
                        torch.cuda.empty_cache()
        
        self.model.config.use_cache = self.use_cache
        if self.is_npu:
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()
        
        prune_ratio = self.check_sparsity()
        self.logger.info(f"sparsity ratio check {prune_ratio:.4f}")
        return self.model
    

class Prune_PANGU:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.nsamples = args.nsamples
        self.device = args.device
        self.sparsity_ratio = args.sparsity_ratio
        self.prune_n = args.prune_n
        self.prune_m = args.prune_m
        self.logger = args.logger
        
        # Check if it's NPU device
        self.is_npu = False
        if 'npu' in str(self.device).lower():
            self.is_npu = True

    def init_model(self):
        self.model.eval()
        self.use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        self.layers = self.model.model.layers

    @classmethod
    def find_layers(cls, module, layers=[nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(cls.find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res

    def check_sparsity(self, tolerance=1e-6):
        self.model.config.use_cache = False
        count = 0
        total_params = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            subset = self.find_layers(layer)
            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W == 0).sum().cpu().item()
                total_params += W.numel()
                sub_count += (W == 0).sum().cpu().item()
                sub_params += W.numel()
            self.logger.info(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
        self.model.config.use_cache = self.use_cache
        error = abs(float(count) / total_params - self.sparsity_ratio)
        if error <= tolerance:
            self.logger.info("Pruning correctly executed")
        else:
            self.logger.info("Pruning not performed correctly")
        return float(count)/total_params
    
    @torch.no_grad()
    def prepare_layer_calibration(self, train_loader, layer_ind=0):
        device = self.model.model.embed_tokens.weight.device
        if device.type == 'cpu':
            if ':' in str(self.device):
                device = torch.device(self.device)
            elif self.is_npu:
                device_id = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0')
                device = torch.device(f'npu:{device_id}')
            else:
                device = torch.device('cuda:0')
            self.model.model.embed_tokens.to(device)
        else:
            device = device
        
        self.logger.info(f"using device to calibrate-->device: {device}")

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros((self.nsamples, self.model.seq_len, self.model.config.hidden_size), dtype=dtype,
                           device=device)
        inps.requires_grad = False
        cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs.get('attention_mask', None)
                cache['position_ids'] = kwargs.get('position_ids', None)
                # Pangu model requires position_embeddings (RoPE's cos/sin tensors)
                cache['position_embeddings'] = kwargs.get('position_embeddings', None)
                raise ValueError

        self.layers[layer_ind] = Catcher(self.layers[layer_ind])
        for batch in train_loader:
            try:
                self.model(batch[0].reshape(-1, self.model.seq_len).to(device))
            except ValueError:
                pass
        self.layers[layer_ind] = self.layers[layer_ind].module
        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        position_embeddings = cache['position_embeddings']
        
        # If position_embeddings is None, try to get from model's rope module
        if position_embeddings is None and hasattr(self.model.model, 'layers') and len(self.model.model.layers) > 0:
            first_layer = self.model.model.layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'rotary_emb'):
                rope = first_layer.self_attn.rotary_emb
                if position_ids is not None:
                    try:
                        cos, sin = rope(position_ids)
                        position_embeddings = (cos, sin)
                        self.logger.info("Computed position_embeddings from RoPE module in Prune_PANGU calibration")
                    except Exception as e:
                        self.logger.warning(f"Failed to compute position_embeddings from RoPE: {e}")
        
        self.model.config.use_cache = self.use_cache
        if self.args.free:
            self.model.model.embed_tokens.to("cpu")
        
        if self.is_npu:
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()
        
        return inps, outs, attention_mask, position_ids, position_embeddings

    def forward_layer_wrapper(self, layer, inps, outs, attention_mask, position_ids, GPT, position_embeddings=None):
        subset = self.find_layers(layer)
        gpts = {}
        for name in subset:
            gpts[name] = GPT(self.args, subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        # If position_embeddings is None, try to get from layer's rope module
        if position_embeddings is None and hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
            rope = layer.self_attn.rotary_emb
            if position_ids is not None:
                try:
                    cos, sin = rope(position_ids)
                    position_embeddings = (cos, sin)
                except Exception as e:
                    self.logger.warning(f"Failed to compute position_embeddings from layer RoPE: {e}")
        
        for j in range(inps.shape[0]):
            with torch.no_grad():
                # Pangu model requires passing position_embeddings parameter (RoPE's cos/sin tensors)
                if position_embeddings is not None:
                    if isinstance(position_embeddings, tuple):
                        cos, sin = position_embeddings
                        cos = cos.to(inps.device) if hasattr(cos, 'to') else cos
                        sin = sin.to(inps.device) if hasattr(sin, 'to') else sin
                        current_position_embeddings = (cos, sin)
                    else:
                        current_position_embeddings = position_embeddings
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                   position_ids=position_ids, position_embeddings=current_position_embeddings)[0]
                else:
                    try:
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                    except TypeError as e:
                        self.logger.error(f"Layer requires position_embeddings but it's None: {e}")
                        raise ValueError("Pangu model requires position_embeddings (RoPE cos/sin tensors). "
                                       "Please ensure the model forward pass provides them.") from e
        for h in handles:
            h.remove()
        return subset, gpts

    @timeit
    def prune_layer_weight(self, subset, gpts):
        for i,name in enumerate(subset):
            if self.args.prune_method == 'sparsegpt':
                self.logger.info(f"pruning {name} by SparseGPT")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m,
                                       blocksize=128, percdamp=.01)
                gpts[name].free()
            elif self.args.prune_method == 'wanda':
                self.logger.info(f"pruning {name} by Wanda")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m)
                gpts[name].free()
            elif self.args.prune_method == 'pruner-zero':
                self.logger.info(f"pruning {name} by Pruner-Zero")
                indexed_name = f'{name}_{self.index_layer}'
                gradients = self.gradients_l2[indexed_name]
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m, gradients, engine=self.engine)
                gpts[name].free()
            elif self.args.prune_method == 'admm-grad':
                self.logger.info(f"pruning {name} by ADMM-Grad")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m, percdamp=.1, iterative_prune=15, iters=20, per_out=False)
                gpts[name].free()
            else:
                raise NotImplementedError
            
            if self.is_npu:
                torch.npu.empty_cache()
            else:
                torch.cuda.empty_cache()

    @timeit
    def prune_llm(self, train_loader):
        self.init_model()
        inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)
        if self.args.prune_method == 'pruner-zero':
            self.logger.info("you must loading model gradient for pruner-zero")
            self.gradients_l2 = self.args.gradients_l2
            self.engine = self.args.engine
        
        for i in trange(len(self.layers), desc='Pruning Processing'):
            layer = self.layers[i]
            self.index_layer = f'layer_{i}'
            
            if hasattr(self.model, 'hf_device_map') and f"model.layers.{i}" in self.model.hf_device_map:
                dev = self.model.hf_device_map[f"model.layers.{i}"]
                if attention_mask:
                    inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                else:
                    inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)
                # Handle position_embeddings device
                if position_embeddings is not None and isinstance(position_embeddings, tuple):
                    cos, sin = position_embeddings
                    cos = cos.to(dev) if hasattr(cos, 'to') else cos
                    sin = sin.to(dev) if hasattr(sin, 'to') else sin
                    position_embeddings = (cos, sin)
            elif hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj') and layer.self_attn.q_proj.weight.device.type == 'cpu':
                if ':' in str(self.device):
                    dev = torch.device(self.device)
                elif self.is_npu:
                    device_id = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0')
                    dev = torch.device(f'npu:{device_id}')
                else:
                    dev = torch.device('cuda:0')
                layer.to(dev)
                if attention_mask:
                    inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                else:
                    inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)
                # Handle position_embeddings device
                if position_embeddings is not None and isinstance(position_embeddings, tuple):
                    cos, sin = position_embeddings
                    cos = cos.to(dev) if hasattr(cos, 'to') else cos
                    sin = sin.to(dev) if hasattr(sin, 'to') else sin
                    position_embeddings = (cos, sin)
            else:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj'):
                    dev = layer.self_attn.q_proj.weight.device
                else:
                    if ':' in str(self.device):
                        dev = torch.device(self.device)
                    elif self.is_npu:
                        device_id = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0')
                        dev = torch.device(f'npu:{device_id}')
                    else:
                        dev = torch.device('cuda:0')

            start = time.time()
            # 1. forward layer wrapper
            if self.args.prune_method == 'sparsegpt':
                GPT = SparseGPT
            elif self.args.prune_method == 'wanda':
                GPT = Wanda
            elif self.args.prune_method == 'pruner-zero':
                GPT = PrunerZero
            elif self.args.prune_method == 'admm-grad':
                GPT = AdmmGrad
            else:
                raise NotImplementedError

            subset, gpts = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids, GPT, position_embeddings)
            
            # 2. pruning weight
            self.prune_layer_weight(subset, gpts)
            
            for j in range(self.nsamples):
                with torch.no_grad():
                    # Pass position_embeddings
                    if position_embeddings is not None:
                        if isinstance(position_embeddings, tuple):
                            cos, sin = position_embeddings
                            cos = cos.to(dev) if hasattr(cos, 'to') else cos
                            sin = sin.to(dev) if hasattr(sin, 'to') else sin
                            current_pe = (cos, sin)
                        else:
                            current_pe = position_embeddings
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                       position_ids=position_ids, position_embeddings=current_pe)[0]
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # update next layer inputs
            self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
            inps, outs = outs, inps
            del layer, subset, gpts
            gc.collect()
            if self.is_npu:
                torch.npu.empty_cache()
            else:
                torch.cuda.empty_cache()
            if self.args.free:
                self.layers[i].to("cpu")
                if self.is_npu:
                    torch.npu.empty_cache()
                else:
                    torch.cuda.empty_cache()
        
        self.model.config.use_cache = self.use_cache
        if self.is_npu:
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()
        
        prune_ratio = self.check_sparsity()
        self.logger.info(f"sparsity ratio check {prune_ratio:.4f}")

