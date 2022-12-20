import jax
from jax import numpy as jnp, random
from flax import linen as nn
import einops as ein

class ConvSSC(nn.Module):
    
    case_count: int
    features: int
    kernel_size: int
    strides: int = 1
    padding: str = 'SAME'
    
    def __init__(self, *args, **kwargs):
        self.case_matcher = nn.Conv(self.case_count, self.kernel_size, strides=self.strides, padding=self.padding,
                                    name='case_matcher', *args, **kwargs)
        self.cases = [nn.Conv(self.features, self.kernel_size, strides=self.strides, padding=self.padding,
                              name=f'case_{i}', *args, **kwargs) for i in range(self.case_count)]
        
    def __call__(self, x_batch):
        case_scores = self.case_matcher(x_batch)
        case_vals = ein.rearrange(
            jnp.array([case(x_batch) for case in self.cases]),
            's b h w c -> b h w s c'
        )
        case_weights = nn.softmax(case_scores, axis=-1) # TODO: check which axis is the pixel axis
        weighted_vals = ein.repeat(case_weights, 'b h w s -> b h w s features', features=self.features) * case_vals
        ret_vals = ein.reduce(weighted_vals, 'b h w s c -> b h w c', 'sum')
        
        return ret_vals
