from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from flax_lora.config import PeftConfigMixin

@dataclass
class LoraConfig(PeftConfigMixin):
    rank: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    
    def __post_init__(self):
        self.peft_type = "LORA"
