from functools import partial
from typing import List, Tuple

import flax
import flax.linen as nn
import jax
from flax.core.frozen_dict import FrozenDict

import enum

CONFIG_NAME = "adapter_config.json"

class PeftType(str, enum.Enum):
    """Enum class for the different types of adapters in PEFT."""

    PROMPT_TUNING = "PROMPT_TUNING"
    MULTITASK_PROMPT_TUNING = "MULTITASK_PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"
    LOHA = "LOHA"
    LOKR = "LOKR"
    OFT = "OFT"


class TaskType(str, enum.Enum):
    """Enum class for the different types of tasks supported by PEFT."""

    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"
    QUESTION_ANS = "QUESTION_ANS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"


def merge_lora_params(base_params: dict | FrozenDict, lora_update_params: dict | FrozenDict) -> dict | FrozenDict:
    """
    Merge the lora_update_params with the original base params
    """
    if isinstance(base_params, FrozenDict):
        return jax.tree_map(
            lambda bp, lp: bp + lp if lp is not None else bp, base_params.unfreeze(), lora_update_params
        )
    return jax.tree_map(lambda bp, lp: bp + lp if lp is not None else bp, base_params, lora_update_params)


def is_tuple(*args, **kwargs):
    """
    Check if the first argument is a tuple
    Used to determine if a leaf of a pytree is a tuple
    """
    return isinstance(args[0], tuple)


def extract_layer_from_iterable(iterable, name):
    for elem in iterable:
        if elem.name == name:
            return elem
    return 
def get_module_by_name(module:nn.Module, name:str):
    if hasattr(module, name):
        return getattr(module, name)
    else:
        iterable_keys = [k for k,v in module.__dict__.items() if isinstance(v, tuple)]
        for key in iterable_keys:
            # print(f"{key=}")
            iterable = getattr(module, key)
            output =  extract_layer_from_iterable(iterable, name)
            
            if output:
                return output
                
def module_iterable(item : nn.Module | Tuple | List) -> bool:
    if isinstance(item, nn.Module):
        return True
    elif isinstance(item, tuple) or isinstance(item, list):
        return all(isinstance(elem, nn.Module) for elem in item)
    else:
        return False
    
def get_layer_type(binded_module:nn.Module, layer:Tuple[str, ...]):
    layer_level = binded_module
    for level in layer:
        # print(f"{level=}")
        layer_level = get_module_by_name(layer_level, str(level))
    return type(layer_level)

def get_model_type_pytree(module:nn.Module, params:dict|FrozenDict):
    flat_params = flax.traverse_util.flatten_dict(params)
    layers = list(set([elem[:-1] for elem in flat_params.keys()]))
    binded_module = module.bind(params)
    layer_types = list(map(partial(get_layer_type, binded_module), layers))
    layer_types_dict = dict(zip(layers, layer_types))
    return flax.traverse_util.unflatten_dict(layer_types_dict)