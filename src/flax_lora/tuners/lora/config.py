from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union
from functools import partial
import re
from ...config import PeftConfigMixin
from ...utils import PeftType, TaskType

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
    bias: Literal["none", "lora_only", "all"] = field(default="none", metadata={"help": "Lora bias"})
    
    def __post_init__(self) -> None:
        self.peft_type = PeftType.LORA
        if isinstance(self.target_modules, str):
            self._target_modules = [self.target_modules]
        else:
            self._target_modules = self.target_modules
            
        self.patterns = [r'(.+\.)?{target}(\..+)?\b'.format(target=target) for target in self._target_modules]
        
    def match_key(self, key: List[str]):
        last_key = key[-1]
        if last_key == "bias" and self.bias =="none":
            return False
        if last_key == "bias" and self.bias =="all":
            return True
        # pattern = r'(.+\.)?{target}(\..+)?\b'
        dot_joint_key = ".".join(key)
        return any(re.match(pattern, dot_joint_key) for pattern in self.patterns)

        
