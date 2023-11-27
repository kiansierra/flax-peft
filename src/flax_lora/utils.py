import jax
from flax.core.frozen_dict import FrozenDict


def merge_lora_params(base_params: dict | FrozenDict, lora_update_params: dict | FrozenDict) -> dict | FrozenDict:
    """
    Merge the lora_update_params with the original base params
    """
    if isinstance(base_params, FrozenDict):
        return jax.tree_map(
            lambda bp, lp: bp + lp if lp is not None else bp, base_params.unfreeze(), lora_update_params
        )
    return jax.tree_map(lambda bp, lp: bp + lp if lp is not None else bp, base_params, lora_update_params)


def is_tuple(*args, **kwargs):
    """
    Check if the first argument is a tuple
    Used to determine if a leaf of a pytree is a tuple
    """
    return isinstance(args[0], tuple)
