import re
from typing import List, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from collections import defaultdict
from .config import LoraConfig


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
        deterministic = kwargs.pop("deterministic", False)
        dropout_rng = kwargs.pop("dropout_rng", None)
        return self.dropout(self.lora_a, deterministic=deterministic, rng=dropout_rng) @ self.lora_b * self.scaling
    
    
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
        dropout_rng = kwargs.pop("dropout_rng", None)
        deterministic = kwargs.pop("deterministic", False)
        
        return self.dropout(self.lora_a, deterministic=deterministic, rng=dropout_rng) @ self.lora_b * self.scaling
        
        
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
        deterministic = kwargs.pop("deterministic", False)
        dropout_rng = kwargs.pop("dropout_rng", None)
        return self.dropout(self.lora_a, deterministic=deterministic, rng=dropout_rng) @ self.lora_b * self.scaling
    
class LoraDefaultLayer(nn.Module):
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
        deterministic = kwargs.pop("deterministic", False)
        dropout_rng = kwargs.pop("dropout_rng", None)
        return self.dropout(self.lora_a, deterministic=deterministic, rng=dropout_rng) @ self.lora_b * self.scaling
                
                
LAYER_MAPPING = defaultdict(lambda: LoraDefaultLayer)
LAYER_MAPPING[nn.Embed] = LoraEmbedding
LAYER_MAPPING[nn.Dense] = LoraDense
LAYER_MAPPING[nn.Conv] = LoraConv
