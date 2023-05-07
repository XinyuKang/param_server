import ray
import torch
import numpy as np


@ray.remote
class ParameterServer(object):
    def __init__(self, keys, values):
        self.weights = dict(zip(keys, values))

    def apply_gradients(self, keys, lr, *values):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*values)
        ]

        idx = 0
        for key, value in zip(keys, summed_gradients):
            self.weights[key] -= lr * torch.from_numpy(summed_gradients[idx])
            idx += 1

        return [self.weights[key] for key in keys]

    def add_weight(self, key, value):
        self.weights[key] = value

    def get_weights(self, keys):
        return [self.weights[key] for key in keys]
