from typing import Dict
from flax_lora.config import PeftConfigMixin
from flax_lora.tuners.lora.config import LoraConfig

PEFT_TYPE_TO_CONFIG_MAPPING: Dict[str, PeftConfigMixin] = {

    "LORA": LoraConfig,

}