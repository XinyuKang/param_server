import ray
import torch
import torch.nn as nn
import numpy as np

import models
import data_loader
from consistent_hashing import ConsistentHash


@ray.remote
class ParameterServer(object):
    """Parameter Server class

    Attributes:
        weights: a dictionary of weights
    """

    def __init__(self, keys, values):
        """Initializes a parameter server.

        Args:
            keys: a list of keys
            values: a list of values
        """
        self.weights = dict(zip(keys, values))

    def apply_gradients(self, keys, lr, *values):
        """Applies gradients to the weights.

        Args:
            keys: a list of keys
            lr: a learning rate
            values: a list of values

        Returns:
            a list of updated weights
        """
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*values)
        ]

        idx = 0
        for key, value in zip(keys, summed_gradients):
            self.weights[key] -= lr * torch.from_numpy(summed_gradients[idx])
            idx += 1

        return [self.weights[key] for key in keys]

    def add_weight(self, key, value):
        """Adds a weight to the dictionary.

        Args:
            key: a key
            value: a value
        """
        self.weights[key] = value

    def get_weights(self, keys):
        """Gets weights from the dictionary.

        Args:
            keys: a list of keys

        Returns:
            a list of weights
        """
        return [self.weights[key] for key in keys]


@ray.remote
class Worker(object):
    """Worker class

    Attributes:
        model: a model
        data_iterator: an iterator for the data
        keys: a list of keys
        key_set: a set of keys
    """

    def __init__(self, keys):
        """Initializes a worker.

        Args:
            keys: a list of keys
        """
        self.model = models.LinearNet()
        self.data_iterator = iter(data_loader.get_data_loader()[0])
        self.keys = keys
        self.key_set = set(self.keys)
        for key, value in dict(self.model.named_parameters()).items():
            if key not in self.key_set:
                value.requires_grad = False

    def update_weights(self, keys, *weights):
        """Updates the weights of the model.

        Args:
            keys: a list of keys
            weights: a list of weights
        """
        self.model.set_weights(keys, weights)

    def update_trainable(self, keys):
        """Updates the trainable weights of the model.

        Args:
            keys: a list of keys
        """
        self.keys = keys
        self.key_set = set(self.keys)
        for key, value in dict(self.model.named_parameters()).items():
            if key in self.key_set:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def compute_gradients(self):
        """Computes gradients for the model.

        Returns:
            a list of gradients

        Raises:
            StopIteration: when the epoch ends, start a new epoch
        """
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(data_loader.get_data_loader()[0])
            data, target = next(self.data_iterator)

        self.model.zero_grad()
        output = self.model(data)
        loss = nn.BCEWithLogitsLoss()(output, target.float())
        loss.backward()

        return self.model.get_gradients(self.keys)


def Scheduler(num_servers, num_workers, hashes_per_server=50):
    """Creates a scheduler.

    Args:
        num_servers: number of servers
        num_workers: number of workers
        hashes_per_server: number of hashes per server
    """
    model = models.LinearNet()
    key_values = model.get_weights()
    keys = np.array(list(key_values.keys()))
    values = [key_values[key] for key in keys]

    key_indices = {key: x for x, key in enumerate(keys)}

    # distributing weights across servers - do this using consistency hashing
    server_ids = ["server" + str(ind) for ind in range(num_servers)]
    hasher = ConsistentHash(keys, server_ids, hashes_per_server)
    servers = [
        ParameterServer.remote(
            keys[[key_indices[key] for key in hasher.get_keys_per_node()[serv]]],
            [values[key_indices[key]] for key in hasher.get_keys_per_node()[serv]],
        )
        for serv in server_ids
    ]

    # creating equal workers per server
    weight_assignments = hasher.get_keys_per_node()
    workers = [
        [
            Worker.remote(weight_assignments["server" + str(j)])
            for i in range(num_workers)
        ]
        for j in range(num_servers)
    ]

    return (
        hasher,
        servers,
        workers,
        keys,
        model,
        hasher.get_keys_per_node(),
        server_ids.copy(),
    )
