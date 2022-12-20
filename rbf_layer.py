import jax
from jax import numpy as jnp, random, tree_util
from flax import linen as nn

class RBFLayer(nn.Module):
    
    n_proxies: int=4
    output_dim: int=1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        centers = self.param('centers', random.normal, (self.n_proxies, x.shape[-1]))
        spreads = self.param('spreads', lambda key, shape: jnp.ones(shape), (self.n_proxies,))
        
        def compute_output(input: jnp.ndarray):
            # RBF weights
            sq_dis = jnp.sum((input[None] - centers) ** 2, axis=1)
            weights = nn.softmax(-sq_dis * spreads, axis=-1)
            
            # calculate values
            proxy_outputs = jnp.array(
                tree_util.tree_map(lambda proxy: proxy(input), [nn.Dense(self.output_dim) for _ in range(self.n_proxies)])
            )
            
            return jnp.matmul(weights, proxy_outputs)
        
        return jax.vmap(compute_output)(x)