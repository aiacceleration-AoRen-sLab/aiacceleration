import gc
import inspect
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import time
from loguru import logger

def cleanup_memory(verbose=True) -> None:
    """Clear GPU memory by running garbage collection and emptying cache."""
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass
    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))
    memory_before = total_reserved_mem()
    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbose:
            logger.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def _nosplit_for(model):
    # Judge by config / class name
    mt = getattr(getattr(model, "config", None), "model_type", "").lower()
    name = model.__class__.__name__
    if "llama" in mt or "llama" in name:
        return ["LlamaDecoderLayer"]
    if "opt" in mt or "OPTForCausalLM" in name or "OPTDecoderLayer" in name:
        return ["OPTDecoderLayer"]
    if "qwen2" in mt or "Qwen2" in name:
        return ["Qwen2DecoderLayer"]
    # Fallback: don't split decoder layers
    return [n for n in [
        "LlamaDecoderLayer", "OPTDecoderLayer", "Qwen2DecoderLayer"
    ] if any(hasattr(__import__("builtins"), "object") for _ in [0])]

def distribute_model(model, device_map=None) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2/3/Qwen-2."""
    # no_split_module_classes = ['LlamaDecoderLayer']
    no_split_module_classes = _nosplit_for(model)
    max_memory = get_balanced_memory(model, no_split_module_classes=no_split_module_classes)
    if not device_map:
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
    logger.info(f"Using device_map: {device_map} for llm evaluation")
    # —— Key patch: force embed/lm_head on same card ——
    # Find the device of the first decoder layer, put embeddings and lm_head there (or the card you want)
    # try:
    #     # OPT structure: model.model.decoder.layers
    #     dec = getattr(model, "model", None)
    #     if dec is not None:
    #         dec = getattr(dec, "decoder", dec)
    #     first_layer_key = None
    #     for k in device_map.keys():
    #         if "decoder.layers.0" in k or "model.layers.0" in k:
    #             first_layer_key = k
    #             break
    #     first_dev = device_map[first_layer_key] if first_layer_key else next(iter(device_map.values()))

    #     # Different model naming differences: try to be compatible
    #     for key in [
    #         "model.decoder.embed_tokens",
    #         "model.model.embed_tokens",
    #         "model.embed_tokens",
    #     ]:
    #         if key in device_map:
    #             device_map[key] = first_dev
    #     if "lm_head" in device_map:
    #         device_map["lm_head"] = first_dev
    # except Exception:
    #     pass
    start = time.time()
    dispatch_model(model, device_map=device_map, offload_buffers=True, offload_dir="offload", state_dict=model.state_dict())
    cleanup_memory()
    logger.info(f"distribute finish, runing time:{time.time() - start}")