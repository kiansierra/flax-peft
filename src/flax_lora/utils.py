from typing import List, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from .lora_config import LoraConfig


def merge_lora_params(base_params:dict | FrozenDict, lora_update_params: dict | FrozenDict) -> dict | FrozenDict:
    """
    Merge the lora_update_params with the original params
    """
    if isinstance(base_params, FrozenDict):
        return jax.tree_map(lambda  p, l: p+l if l is not None else p,  base_params.unfreeze(), lora_update_params)
    return jax.tree_map(lambda  p, l: p+l if l is not None else p,  base_params, lora_update_params)

def is_tuple(*args, **kwargs):
    return isinstance(args[0], tuple)