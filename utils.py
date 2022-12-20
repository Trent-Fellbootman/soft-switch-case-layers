from jax import tree_util, numpy as jnp
from optax._src.base import PyTree

def count_parameters(parameters: PyTree):
    count = 0
    def add_count(array):
        nonlocal count
        count += jnp.size(array)
    tree_util.tree_map(add_count, parameters)
    
    return count