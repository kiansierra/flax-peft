from dataclasses import dataclass, field
from typing import List


@dataclass
class LoraConfig:
    rank : int 
    lora_alpha:int 
    target_modules: List[str] = field()
    lora_dropout:float =0.0