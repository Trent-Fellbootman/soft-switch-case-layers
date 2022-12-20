from flax import linen as nn
import jax
from jax import numpy as np, random
from base import ModelInstance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches
from tqdm import tqdm
from typing import Callable
import optax

from utils import count_parameters

def test_model(model: nn.Module, X, y, loss_fn: Callable, need_vmap: bool=True, optimizer: optax.GradientTransformation=optax.adam(1e-3),
               num_epochs: int=1000, batch_size: int=32, test_size: float=0.2, key: random.KeyArray=random.PRNGKey(0)):
    instance = ModelInstance(model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    batch_indices = list(gen_batches(X_train.shape[0], batch_size))
    
    instance.initialize(X_train)
    print(f'parameter count: {count_parameters(instance.parameters_)}')
    instance.compile(loss_fn, need_vmap)
    instance.attach_optimizer(optimizer)
    
    epochs = tqdm(range(num_epochs))
    history = {'train_loss': [], 'val_loss': []}
    for epoch in epochs:
        epochs.set_description(f'epoch {epoch}: val_loss: {instance.compute_loss(X_test, y_test): .2e}')
        
        key, new_key = random.split(key)
        X_train = random.permutation(new_key, X_train)
        y_train = random.permutation(new_key, y_train)

        iterations = tqdm(enumerate(range(0, X_train.shape[0] - batch_size + 1, batch_size)))
        for i, start in iterations:
            end = start + batch_size
            train_loss = instance.step(X_train[start:end], y_train[start:end])
            val_loss = instance.compute_loss(X_test, y_test)
            iterations.set_description(f'iteration {i}: train_loss: {train_loss: .2e}, val_loss: {val_loss: .2e}')

            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
    
    plt.figure(figsize=(10, 6))
    for key, value in history.items():
        plt.plot(range(len(value)), value, label=key)
    plt.legend()
    plt.show()
    
    return history
    