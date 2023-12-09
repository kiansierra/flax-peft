import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import FlaxBertForMaskedLM, FlaxBertModel

from flax_lora import LoraConfig, build_lora_model


def test_bert_targets() -> None:
    model = FlaxBertForMaskedLM.from_pretrained('prajjwal1/bert-mini', revision='refs/pr/3')
    module, base_params = model.module, model.params
    lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['word_embeddings'])
    _, lora_params, _ = build_lora_model(module, lora_config, base_params)
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    assert all("word_embeddings" in k for k in flat_lora_params.keys()), "Lora params should have 'word_embedding' in their keys"
    
def test_bert_mlm() -> None:
    model = FlaxBertForMaskedLM.from_pretrained('prajjwal1/bert-mini', revision='refs/pr/3')
    module, base_params = model.module, model.params
    lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['word_embeddings'])
    _, lora_params, _  = build_lora_model(module, lora_config, base_params)
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    print(f"{flat_lora_params.keys()=}")
    assert all("word_embeddings" in k for k in flat_lora_params.keys()), "Lora params should have 'word_embedding' in their keys"
    
    
def test_bert_lora_mappings_embeddings() -> None:
    model = FlaxBertModel.from_pretrained('prajjwal1/bert-mini', revision='refs/pr/3')
    module, base_params = model.module, model.params
    lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['word_embeddings'])
    _, lora_params, _  = build_lora_model(module, lora_config, base_params)
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    assert len(flat_lora_params) == 2, "Lora_a word_embeddings and Lora_b word_embeddings should be the only params"
    
    
def test_bert_lora_mappings_qk() -> None:
    model = FlaxBertModel.from_pretrained('prajjwal1/bert-mini', revision='refs/pr/3')
    module, base_params = model.module, model.params
    lora_config = LoraConfig(rank=16, lora_alpha=2, target_modules=['query', "key"])
    _, lora_params, _  = build_lora_model(module, lora_config, base_params)
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    assert len(flat_lora_params) == 2 * 2 * model.config.num_hidden_layers, "2 lora layers, times 2 for q and k times the number of layers"
    
    
def test_conv() -> None:
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
    _, lora_params, _  = build_lora_model(module, lora_config, base_params['params'])
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    assert len(flat_lora_params) == 2 + 2, "2 lora layers for conv non for conv2 and both bias"
    
    lora_config = LoraConfig(rank=4, lora_alpha=2, target_modules=['conv'], bias="lora_only")
    _, lora_params, _  = build_lora_model(module, lora_config, base_params['params'])
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    assert len(flat_lora_params) == 2 + 1, "2 lora layers for conv non for conv2 and only conv bias"
    

    lora_config = LoraConfig(rank=4, lora_alpha=2, target_modules=['conv'], bias="none")
    _, lora_params, _  = build_lora_model(module, lora_config, base_params['params'])
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    assert len(flat_lora_params) == 2, "2 lora layers for conv non for conv2"
    
    lora_config = LoraConfig(rank=4, lora_alpha=2, target_modules=['conv\d*'], bias="none")
    _, lora_params, _  = build_lora_model(module, lora_config, base_params['params'])
    flat_lora_params = flax.traverse_util.flatten_dict(lora_params)
    assert len(flat_lora_params) == 4, "2 lora layers for conv and 2 for conv2"
    

    
    