import flax
import jax
import jax.numpy as jnp
from transformers import BertTokenizerFast, FlaxBertForMaskedLM, FlaxBertModel

from flax_lora import LoraConfig, LoraWrapper, build_lora_model


def test_bert() -> None:
    key = jax.random.PRNGKey(0)
    model = FlaxBertForMaskedLM.from_pretrained('prajjwal1/bert-mini', revision='refs/pr/3')
    module, base_params = model.module, model.params
    lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['word_embedding'])
    lora_module = build_lora_model(module, lora_config, base_params)
    inputs = dict(input_ids=jnp.ones((1, 1), dtype=jnp.int32),
                  attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
                  token_type_ids=None, position_ids=None, head_mask=None)
    lora_params = lora_module.init(key, base_params, **inputs)
    model_outputs = module.apply({'params': base_params}, **inputs)
    lora_outputs = lora_module.apply(lora_params, base_params, **inputs)
    assert jnp.allclose(model_outputs.logits, lora_outputs.logits, atol=1e-8), "Lora outputs should be close to the base model outputs"
    
    
def test_bert_targets() -> None:
    key = jax.random.PRNGKey(0)
    model = FlaxBertForMaskedLM.from_pretrained('prajjwal1/bert-mini', revision='refs/pr/3')
    module, base_params = model.module, model.params
    lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['word_embeddings'])
    lora_module = build_lora_model(module, lora_config, base_params)
    inputs = dict(input_ids=jnp.ones((1, 1), dtype=jnp.int32),
                  attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
                  token_type_ids=None, position_ids=None, head_mask=None)
    lora_params = lora_module.init(key, base_params, **inputs)
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    print(f"{flat_lora_params.keys()=}")
    assert all("word_embeddings" in k for k in flat_lora_params.keys()), "Lora params should have 'word_embedding' in their keys"
    
    