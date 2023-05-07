import ray
import models
import data_loader
import torch.nn as nn
import data_loader


@ray.remote
class DataWorker(object):
    def __init__(self, keys):
        self.model = models.LinearNet()
        self.data_iterator = iter(data_loader.get_data_loader()[0])
        self.keys = keys
        self.key_set = set(self.keys)
        for key, value in dict(self.model.named_parameters()).items():
            if key not in self.key_set:
                value.requires_grad = False

    def update_weights(self, keys, *weights):
        self.model.set_weights(keys, weights)

    def update_trainable(self, keys):
        self.keys = keys
        self.key_set = set(self.keys)
        for key, value in dict(self.model.named_parameters()).items():
            if key in self.key_set:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def compute_gradients(self):
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
