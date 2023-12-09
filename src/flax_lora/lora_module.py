import re
from collections import defaultdict
from typing import List, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from .tuners.lora import LoraConfig, LoraModule, LoraWrapper
from .utils import get_model_type_pytree, is_tuple, merge_lora_params, GeneralDict


def select_target_modules(lora_config: LoraConfig, shape_tree:GeneralDict) -> GeneralDict:
    """
    Make a dictionary of target modules
    """
    # Generate a flat pytree with the lora_config in each node
    config_dict = flax.traverse_util.flatten_dict(
        jax.tree_util.tree_map(lambda x: lora_config, shape_tree, is_leaf=is_tuple)
    )
    # Generate a flat pytree with the weight shape in each node
    shape_dict = flax.traverse_util.flatten_dict(shape_tree)
    # Merge the shape and config dicts so we can initialize the LoraModule
    # pick first config since it gets duplicated for each shape dimension
    flat_dict = {
        k: (v, shape_dict[k]) for k, v in config_dict.items() if lora_config.match_key(k)
    }
    lora_dict = flax.traverse_util.unflatten_dict(flat_dict)
    return lora_dict


def build_lora_model(model: nn.Module, lora_config: LoraConfig, params: GeneralDict):
    shape_tree = jax.tree_util.tree_map(lambda x: x.shape, params)
    targets_tree = select_target_modules(lora_config, shape_tree)
    selected_keys = flax.traverse_util.flatten_dict(targets_tree).keys()
    flat_params = flax.traverse_util.flatten_dict(params)
    selected_params = {k: v for k, v in flat_params.items() if k in selected_keys}
    selected_params = flax.traverse_util.unflatten_dict(selected_params)
    type_tree = get_model_type_pytree(model, selected_params)
    targets_tree = flax.traverse_util.flatten_dict(targets_tree)
    type_tree = flax.traverse_util.flatten_dict(type_tree)  
    targets_tree = {k: (*v, type_tree[k[:-1]]) for k, v in targets_tree.items()}
    targets_tree = flax.traverse_util.unflatten_dict(targets_tree)
    return LoraWrapper(model, shape_tree, targets_tree)
