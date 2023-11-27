# %%
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from dataclasses import dataclass, field

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.linen.dtypes import promote_dtype
from jax import lax
from transformers import BertTokenizerFast, FlaxBertModel

from loras import LoraConfig, LoraWrapper

print(f"{jax.__version__=}")
print(f"{optax.__version__=}")


# %%
key = jax.random.PRNGKey(0)
model = FlaxBertModel.from_pretrained('bert-base-uncased')

# %%
optimizer = optax.adamw(learning_rate=1e-3)

# %%
lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['embeddings'])
lora_config

# %%
lora = LoraWrapper(model.module, lora_config, model.params)

# %%
lora_params = lora.init(key, jnp.ones((1, 1), dtype=jnp.int32), jnp.ones((1, 1), dtype=jnp.int32))

# %%
