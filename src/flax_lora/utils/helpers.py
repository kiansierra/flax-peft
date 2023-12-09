import jax
import flax
import flax.linen as nn
from functools import partial
from typing import List, Tuple
from flax.core.frozen_dict import FrozenDict

GeneralDict = dict | FrozenDict
EmbeddingsShape = Tuple[int, int]
GeneralShape = Tuple[int, ...]

def is_nullable_array(*args, **kwargs):
    """
    Check if the first argument is a jax array
    Used to determine if a leaf of a pytree is a jax array
    """
    return isinstance(args[0], jax.Array) or args[0] is None

def merge_lora_params(base_params: dict | FrozenDict, lora_update_params: dict | FrozenDict) -> dict | FrozenDict:
    """
    Merge the lora_update_params with the original base params
    """
    def sum_nullable_params(bp, lp):
        if lp is not None and bp is not None:
            return bp + lp
        elif lp is None and bp is not None:
            return bp
        elif lp is not None and bp is None:
            return lp
        else:
            raise ValueError("Both base_params and lora_update_params are None")
        
    
    if isinstance(base_params, FrozenDict):
        return jax.tree_map(sum_nullable_params, base_params.unfreeze(), lora_update_params, is_leaf=is_nullable_array
        )
    return jax.tree_map(sum_nullable_params, base_params, lora_update_params, is_leaf=is_nullable_array)


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