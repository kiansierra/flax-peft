import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import FlaxBertForMaskedLM, FlaxBertModel

from flax_lora import LoraConfig, build_lora_model


def test_bert_first_forward() -> None:
    model = FlaxBertModel.from_pretrained('prajjwal1/bert-mini', revision='refs/pr/3')
    module, base_params = model.module, model.params
    lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['word_embeddings'])
    lora_module, lora_params, filtered_base_params = build_lora_model(module, lora_config, base_params)
    inputs = dict(input_ids=jnp.ones((1, 1), dtype=jnp.int32),
                  attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
                  token_type_ids=None, position_ids=None, head_mask=None)
    model_outputs = module.apply({'params': base_params}, **inputs)
    lora_outputs = lora_module.apply(lora_params, filtered_base_params, **inputs)
    assert jnp.allclose(model_outputs.last_hidden_state, lora_outputs.last_hidden_state, atol=1e-8), "Lora outputs should be close to the base model outputs"
    
    lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['word_embeddings'], bias="all")
    lora_module, lora_params, filtered_base_params = build_lora_model(module, lora_config, base_params)
    inputs = dict(input_ids=jnp.ones((1, 1), dtype=jnp.int32),
                  attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
                  token_type_ids=None, position_ids=None, head_mask=None)
    model_outputs = module.apply({'params': base_params}, **inputs)
    lora_outputs = lora_module.apply(lora_params, filtered_base_params, **inputs)
    assert jnp.allclose(model_outputs.last_hidden_state, lora_outputs.last_hidden_state, atol=1e-8), "Lora outputs should be close to the base model outputs"
    
    
def test_conv_forward() -> None:
    key = jax.random.PRNGKey(0)
    class ConvModule(nn.Module):
        features:int
        def setup(self) -> None:
            self.conv = nn.Conv(features=self.features, kernel_size=(3, 3), bias_init=nn.initializers.ones)
            self.conv2 = nn.Conv(features=self.features, kernel_size=(3, 3), bias_init=nn.initializers.ones)
            return super().setup()
        
        def __call__(self, x):
            return self.conv2(self.conv(x))
    module = ConvModule(features=32)
    inputs = jnp.ones((1, 3, 256, 256))
    base_params = module.init(key, inputs)
    
    lora_config = LoraConfig(rank=4, lora_alpha=2, target_modules=['conv'], bias="all")
    lora_module, lora_params, base_updated_prams  = build_lora_model(module, lora_config, base_params['params'])
    lora_outputs = lora_module.apply(lora_params, base_updated_prams, lora_deterministic=True, x=inputs)
    model_outputs = module.apply(base_params, inputs)
    assert jnp.allclose(model_outputs, lora_outputs, atol=1e-8), "Lora outputs should be close to the base model outputs"
    
    lora_config = LoraConfig(rank=4, lora_alpha=2, target_modules=['conv'], bias="none")
    lora_module, lora_params, base_updated_prams  = build_lora_model(module, lora_config, base_params['params'])
    lora_outputs = lora_module.apply(lora_params, base_updated_prams, lora_deterministic=True, x=inputs)
    model_outputs = module.apply(base_params, inputs)
    assert jnp.allclose(model_outputs, lora_outputs, atol=1e-8), "Lora outputs should be close to the base model outputs"
    
    lora_config = LoraConfig(rank=4, lora_alpha=2, target_modules=['conv'], bias="lora_only")
    lora_module, lora_params, base_updated_prams  = build_lora_model(module, lora_config, base_params['params'])
    lora_outputs = lora_module.apply(lora_params, base_updated_prams, lora_deterministic=True, x=inputs)
    model_outputs = module.apply(base_params, inputs)
    assert jnp.allclose(model_outputs, lora_outputs, atol=1e-8), "Lora outputs should be close to the base model outputs"
        
