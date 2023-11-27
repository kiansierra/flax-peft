from typing import List, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from .lora_config import LoraConfig


class LoraModule(nn.Module):
    
    lora_dict: dict | Tuple[LoraConfig, jnp.shape]
    
    def setup(self):
        if isinstance(self.lora_dict, tuple):
            lora_config = self.lora_dict[0]
            weight_shape = self.lora_dict[1]
            self.lora_a = self.param('lora_a',
                       lambda rng, shape: jax.random.normal(rng, shape),
                       (weight_shape[0], lora_config.rank))
            
            self.lora_b = self.param('lora_b',
                       lambda rng, shape: jnp.ones(shape),
                       (lora_config.rank, weight_shape[1]))
        else:
            for k, v in self.lora_dict.items():
                setattr(self, k, LoraModule(lora_dict=v, name=k))
    
    def __call__(self, *args, **kwargs):
        if isinstance(self.lora_dict, tuple):
            return self.lora_a @ self.lora_b
        output = {}
        for k in self.lora_dict.keys():
            if k in ('bias', 'scale'): 
                continue
            output[k] = getattr(self, k)(*args, **kwargs)
        return output
    
def match_key(key, target_modules:List[str]):
    return any(target in ".".join(key) for target in target_modules)

class LoraWrapper(nn.Module):
    model: nn.Module
    lora_config: LoraConfig
    params: dict | FrozenDict
    
    def select_target_modules(self):
        """
        Make a dictionary of target modules
        """
        # Generate a flat pytree with the lora_config in each node
        config_dict = flax.traverse_util.flatten_dict(jax.tree_util.tree_map(lambda x: self.lora_config, self.params))
        # Generate a flat pytree with the weight shape in each node
        shape_dict = flax.traverse_util.flatten_dict(jax.tree_util.tree_map(lambda x: x.shape, self.params))
        # Merge the shape and config dicts so we can initialize the LoraModule
        flat_dict= {k: (v, shape_dict[k]) for k, v in config_dict.items() if match_key(k, self.lora_config.target_modules)}
        lora_dict = flax.traverse_util.unflatten_dict(flat_dict)
        return lora_dict
    
    def setup(self):
        self.lora = LoraModule(lora_dict=self.select_target_modules())
        
    def complete_tree(self, lora_output):
        """
        Expand the lora_output pytree to match the shape of the original pytree
        Empty leafs are filled with None
        """
        empty_tree = jax.tree_map(lambda x: None, self.params)
        empty_tree = flax.traverse_util.flatten_dict(empty_tree)
        lora_flat_params = flax.traverse_util.flatten_dict(lora_output)
        empty_tree.update(lora_flat_params)
        lora_fused_params = flax.traverse_util.unflatten_dict(empty_tree)
        return lora_fused_params
    
    def merge_params(self, lora_update_params):
        """
        Merge the lora_update_params with the original params
        """
        if isinstance(self.params, FrozenDict):
            return jax.tree_map(lambda  p, l: p+l if l is not None else p,  self.params.unfreeze(), lora_update_params)
        return jax.tree_map(lambda  p, l: p+l if l is not None else p,  self.params, lora_update_params)
    
    
    def __call__(self, *args, **kwargs):
        lora_output = self.lora(*args, **kwargs)
        lora_update_params = self.complete_tree(lora_output)
        lora_fused_params = self.merge_params(lora_update_params)
        return self.model.apply({'params':lora_fused_params},*args, **kwargs)
    
    