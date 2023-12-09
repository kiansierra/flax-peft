import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax_lora import LoraConfig, build_lora_model
from transformers import FlaxBertForMaskedLM, FlaxBertModel


def test_bert_targets() -> None:
    key = jax.random.PRNGKey(0)
    model = FlaxBertForMaskedLM.from_pretrained('prajjwal1/bert-mini', revision='refs/pr/3')
    module, base_params = model.module, model.params
    lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['word_embeddings'])
    lora_module = build_lora_model(module, lora_config, base_params)
    lora_params = lora_module.init(key, base_params, method=lora_module.delta_weights)
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    assert all("word_embeddings" in k for k in flat_lora_params.keys()), "Lora params should have 'word_embedding' in their keys"
    
def test_bert_mlm() -> None:
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
    
    
def test_bert_lora_mappings_embeddings() -> None:
    key = jax.random.PRNGKey(0)
    model = FlaxBertModel.from_pretrained('prajjwal1/bert-mini', revision='refs/pr/3')
    module, base_params = model.module, model.params
    lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['word_embeddings'])
    lora_module = build_lora_model(module, lora_config, base_params)
    inputs = dict(input_ids=jnp.ones((1, 1), dtype=jnp.int32),
                  attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
                  token_type_ids=None, position_ids=None, head_mask=None)
    lora_params = lora_module.init(key, base_params, **inputs)
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    assert len(flat_lora_params) == 2, "Lora_a word_embeddings and Lora_b word_embeddings should be the only params"
    
    
def test_bert_lora_mappings_qk() -> None:
    key = jax.random.PRNGKey(0)
    model = FlaxBertModel.from_pretrained('prajjwal1/bert-mini', revision='refs/pr/3')
    module, base_params = model.module, model.params
    lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['query', "key"])
    lora_module = build_lora_model(module, lora_config, base_params)
    inputs = dict(input_ids=jnp.ones((1, 1), dtype=jnp.int32),
                  attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
                  token_type_ids=None, position_ids=None, head_mask=None)
    lora_params = lora_module.init(key, base_params, **inputs)
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    assert len(flat_lora_params) == 2 * 2 * model.config.num_hidden_layers, "2 lora layers, times 2 for q and k times the number of layers"
    
    
def test_conv() -> None:
    key = jax.random.PRNGKey(0)
    class ConvModule(nn.Module):
        features:int
        def setup(self) -> None:
            self.conv = nn.Conv(features=self.features, kernel_size=(3, 3))
            self.conv2 = nn.Conv(features=self.features, kernel_size=(3, 3))
            return super().setup()
        
        def __call__(self, x):
            return self.conv2(self.conv(x))
    module = ConvModule(features=32)
    inputs = jnp.ones((1, 3, 256, 256))
    base_params = module.init(key, inputs)
    

    lora_config = LoraConfig(rank=4, lora_alpha=2, target_modules=['conv'])
    lora_module = build_lora_model(module, lora_config, base_params['params'])
    lora_params = lora_module.init(key, base_params['params'], method=lora_module.delta_weights)
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    assert len(flat_lora_params) == 2, "2 lora layers for conv non for conv2"
    
    lora_config = LoraConfig(rank=4, lora_alpha=2, target_modules=['conv\d*'])
    lora_module = build_lora_model(module, lora_config, base_params['params'])
    lora_params = lora_module.init(key, base_params['params'], method=lora_module.delta_weights)
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    assert len(flat_lora_params) == 4, "2 lora layers for conv and 2 for conv2"
    
    