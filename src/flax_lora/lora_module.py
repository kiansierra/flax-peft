import re
from typing import List, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from .lora_config import LoraConfig
from .utils import get_model_type_pytree, is_tuple, merge_lora_params

GeneralDict = dict | FrozenDict
EmbeddingsShape = Tuple[int, int]
GeneralShape = Tuple[int, ...]

class LoraEmbedding(nn.Module):
    lora_config: LoraConfig
    weight_shape: GeneralShape
    
    def setup(self):
        lora_config = self.lora_config
        weight_shape = self.weight_shape
        self.scaling = lora_config.lora_alpha / lora_config.rank
        self.dropout = nn.Dropout(rate=lora_config.lora_dropout) if lora_config.lora_dropout > 0 else lambda x, *args, **kwargs: x
        self.lora_a = self.param(
            "lora_a", lambda rng, shape: jax.random.normal(rng, shape), (weight_shape[0], lora_config.rank)
        )

        self.lora_b = self.param(
            "lora_b", lambda rng, shape: jnp.zeros(shape), (lora_config.rank, weight_shape[1])
        )
    def __call__(self, *args, **kwargs):
        train = kwargs.pop("train", False)
        return self.dropout(self.lora_a, deterministic=not train) @ self.lora_b * self.scaling
    
    
class LoraDense(nn.Module):
    lora_config: LoraConfig
    weight_shape: GeneralShape
    
    def setup(self):
        lora_config = self.lora_config
        weight_shape = self.weight_shape
        self.scaling = lora_config.lora_alpha / lora_config.rank
        self.dropout = nn.Dropout(rate=lora_config.lora_dropout) if lora_config.lora_dropout > 0 else lambda x, *args, **kwargs: x
        self.lora_a = self.param(
            "lora_a", lambda rng, shape: jax.random.normal(rng, shape), (weight_shape[0], lora_config.rank)
        )

        self.lora_b = self.param(
            "lora_b", lambda rng, shape: jnp.zeros(shape), (lora_config.rank, weight_shape[1])
        )
    def __call__(self, *args, **kwargs):
        train = kwargs.pop("train", False)
        return self.dropout(self.lora_a, deterministic=not train) @ self.lora_b * self.scaling
        
        
class LoraConv(nn.Module):
    lora_config: LoraConfig
    weight_shape: GeneralShape
    
    def setup(self):
        lora_config = self.lora_config
        weight_shape = self.weight_shape
        self.scaling = lora_config.lora_alpha / lora_config.rank
        self.dropout = nn.Dropout(rate=lora_config.lora_dropout) if lora_config.lora_dropout > 0 else lambda x, *args, **kwargs: x
        self.lora_a = self.param(
            "lora_a", lambda rng, shape: jax.random.normal(rng, shape), (*weight_shape[:-1], lora_config.rank)
        )

        self.lora_b = self.param(
            "lora_b", lambda rng, shape: jnp.zeros(shape), (lora_config.rank, weight_shape[-1])
        )
    def __call__(self, *args, **kwargs):
        train = kwargs.pop("train", False)
        return self.dropout(self.lora_a, deterministic=not train) @ self.lora_b * self.scaling
                


LAYER_MAPPING = {
    nn.Embed: LoraEmbedding,
    nn.Dense: LoraDense,
    nn.Conv : LoraConv
}


class LoraModule(nn.Module):
    lora_dict: GeneralDict | Tuple[LoraConfig, GeneralShape, type[nn.Module]]

    def setup(self):
        for k, v in self.lora_dict.items():
            if isinstance(v, tuple):
                setattr(self, k, LAYER_MAPPING[v[2]](lora_config=v[0], weight_shape=v[1]))
                continue
            setattr(self, k, LoraModule(lora_dict=v, name=k))

    def __call__(self, *args, **kwargs):
        output = {}
        for k in self.lora_dict.keys():
            if k in ("bias", "scale"):
                continue
            output[k] = getattr(self, k)(*args, **kwargs)
        return output


def match_key(key, target_modules: List[str]):
    pattern = r'(.+\.)?{target}(\..+)?\b'
    return any(re.match(pattern.format(target=target), ".".join(key)) for target in target_modules)


def build_lora_model(model: nn.Module, lora_config: LoraConfig, params: GeneralDict):
    shape_tree = jax.tree_util.tree_map(lambda x: x.shape, params)
    type_tree = get_model_type_pytree(model, params)
    targets_tree = select_target_modules(lora_config, shape_tree, type_tree)
    return LoraWrapper(model, shape_tree, targets_tree)

def select_target_modules(lora_config: LoraConfig, shape_tree:dict, type_tree:dict) -> dict:
    """
    Make a dictionary of target modules
    """
    # Generate a flat pytree with the lora_config in each node
    config_dict = flax.traverse_util.flatten_dict(
        jax.tree_util.tree_map(lambda x: lora_config, shape_tree, is_leaf=is_tuple)
    )
    # Generate a flat pytree with the weight shape in each node
    shape_dict = flax.traverse_util.flatten_dict(shape_tree)
    type_dict = flax.traverse_util.flatten_dict(type_tree)
    # Merge the shape and config dicts so we can initialize the LoraModule
    # pick first config since it gets duplicated for each shape dimension
    flat_dict = {
        k: (v, shape_dict[k], type_dict[k[:-1]]) for k, v in config_dict.items() if match_key(k, lora_config.target_modules)
    }
    lora_dict = flax.traverse_util.unflatten_dict(flat_dict)
    return lora_dict

class LoraWrapper(nn.Module):
    model: nn.Module
    shape_tree: GeneralDict
    lora_target_modules: GeneralDict

    def setup(self) -> None:
        self.lora = LoraModule(lora_dict=self.lora_target_modules)

    def complete_tree(self, lora_output):
        """
        Expand the lora_output pytree to match the shape of the original pytree
        Empty leafs are filled with None
        """
        empty_tree = jax.tree_map(lambda x: None, self.shape_tree, is_leaf=is_tuple)
        empty_tree = flax.traverse_util.flatten_dict(empty_tree)
        lora_flat_params = flax.traverse_util.flatten_dict(lora_output)
        empty_tree.update(lora_flat_params)
        lora_fused_params = flax.traverse_util.unflatten_dict(empty_tree)
        return lora_fused_params

    def __call__(self, base_params, *args, **kwargs):
        lora_output = self.lora(*args, **kwargs)
        lora_update_params = self.complete_tree(lora_output)
        lora_fused_params = merge_lora_params(base_params, lora_update_params)
        return self.model.apply({"params": lora_fused_params}, *args, **kwargs)
    
    def merge(self, base_params, *args, **kwargs):
        lora_output = self.lora(*args, **kwargs)
        lora_update_params = self.complete_tree(lora_output)
        lora_fused_params = merge_lora_params(base_params, lora_update_params)
        return lora_fused_params
    
    def delta_weights(self, *args, **kwargs):
        lora_output = self.lora(*args, **kwargs)
        return lora_output
