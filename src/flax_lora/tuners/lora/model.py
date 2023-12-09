from typing import Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from ...utils import GeneralDict, GeneralShape, is_tuple, merge_lora_params
from .config import LoraConfig
from .layer import LAYER_MAPPING


class LoraWrapper(nn.Module):
    lora_dict: GeneralDict | Tuple[LoraConfig, GeneralShape, type[nn.Module]]

    def setup(self) -> None:
        for k, v in self.lora_dict.items():
            if k in ("bias", "scale"):
                setattr(self, k, self.param(k, lambda rng, shape: jnp.zeros(shape, dtype="float32"), v[1]))
            elif isinstance(v, tuple):
                setattr(self, k, LAYER_MAPPING[v[2]](lora_config=v[0], weight_shape=v[1]))
                continue
            else:
                setattr(self, k, LoraWrapper(lora_dict=v, name=k))

    def __call__(self, *args, **kwargs):
        output = {}
        for k in self.lora_dict.keys():
            if k in ("bias", "scale"):
                output[k] = getattr(self, k)
            else:
                output[k] = getattr(self, k)(*args, **kwargs)
        return output
    


class LoraModule(nn.Module):
    model: nn.Module
    shape_tree: GeneralDict
    lora_target_modules: GeneralDict

    def setup(self) -> None:
        self.lora = LoraWrapper(lora_dict=self.lora_target_modules)

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

    def __call__(self, base_params:GeneralDict,
                 lora_dropout_rng:Optional[jax.random.PRNGKey]=None,
                 lora_deterministic:bool=False,
                 *args, **kwargs):
        lora_output = self.lora(dropout_rng=lora_dropout_rng, deterministic=lora_deterministic)
        lora_update_params = self.complete_tree(lora_output)
        lora_fused_params = merge_lora_params(base_params, lora_update_params)
        return self.model.apply({"params": lora_fused_params}, *args, **kwargs)
    
    def merge(self, base_params:GeneralDict) -> GeneralDict:
        lora_output = self.lora(lora_deterministic=True)
        lora_update_params = self.complete_tree(lora_output)
        lora_fused_params = merge_lora_params(base_params, lora_update_params)
        return lora_fused_params
    
    def delta_weights(self, lora_dropout_rng:Optional[jax.random.PRNGKey]=None,
                      lora_deterministic:bool=True) -> GeneralDict:
        lora_output = self.lora(dropout_rng=lora_dropout_rng, deterministic=lora_deterministic)
        return lora_output