import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import FlaxBertForMaskedLM, FlaxBertModel

from flax_lora import LoraConfig, build_lora_model


def test_bert_first_forward() -> None:
    key = jax.random.PRNGKey(0)
    model = FlaxBertModel.from_pretrained('prajjwal1/bert-mini', revision='refs/pr/3')
    module, base_params = model.module, model.params
    lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['word_embeddings'])
    lora_module = build_lora_model(module, lora_config, base_params)
    lora_params = lora_module.init(key, base_params, method=lora_module.delta_weights)
    inputs = dict(input_ids=jnp.ones((1, 1), dtype=jnp.int32),
                  attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
                  token_type_ids=None, position_ids=None, head_mask=None)
    model_outputs = module.apply({'params': base_params}, **inputs)
    lora_outputs = lora_module.apply(lora_params, base_params, **inputs)
    assert jnp.allclose(model_outputs.last_hidden_state, lora_outputs.last_hidden_state, atol=1e-8), "Lora outputs should be close to the base model outputs"
    
    