from model_ops.resnet import *
import numpy as np
from functools import reduce
import torch.distributed as dist
from nn_train import NN_Trainer


class GradientAccumulator(object):
    """This class is used to aggregate gradients from workers

    Attributes:
        gradient_aggregate_counter: the counter used to count the number of gradients
        gradient_aggregator: the buffer used to aggregate gradients
        model_index_range: the range of model parameters
    """

    def __init__(self, module, world_size):
        """Initializes the instance with basic settings.

        Args:
            module: the network model
            world_size: the number of workers
        """
        super(GradientAccumulator, self).__init__()
        # length of this counter should be the # of fc layers in the network
        self.gradient_aggregate_counter = []
        self.gradient_aggregator = []
        self.model_index_range = []

        for param_idx, param in enumerate(module.parameters()):
            tmp_aggregator = []
            for worker_idx in range(world_size):
                tmp_aggregator.append(torch.zeros(param.size()))
            self.gradient_aggregator.append(tmp_aggregator)
            self.gradient_aggregate_counter.append(0)
            self.model_index_range.append(param_idx)


class Server(NN_Trainer):
    """This class is used to train the model on the parameter server side

    Attributes:
        grad_aggregate_buffer: used to aggregate gradients, the length is the same as the # of fc layers
        world_size: number of workers
        eval_batch_size: the batch size used for evaluation
        cur_step: the current training step
        max_num_step: the maximum number of training steps
        lr: the learning rate used for SGD
        momentum: the momentum used for SGD
    """

    def __init__(self, **kwargs):
        """Initializes the instance with basic settings.

        Args:
            kwargs: the keyword arguments
        """
        # used to aggregate gradients, the length is the same as the # of fc layers
        self.grad_aggregate_buffer = []
        self.world_size = kwargs["world_size"]  # number of workers
        self.eval_batch_size = 1000
        self.cur_step = 1
        self.max_num_step = kwargs["max_num_step"]
        self.lr = kwargs["lr"]
        self.momentum = kwargs["momentum"]
        # self.num_workers = kwargs['world_size']

    def build_model(self, num_classes=10):
        """Builds the model.

        Args:
            num_classes: the number of classes. Default: 10
        """
        super().build_model(num_classes)
        # self.network = ResNet18(num_classes)
        # self.optimizer = SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        # collect gradients from workers
        self.grad_accumulator = GradientAccumulator(self.network, self.world_size)

        # initialize model shapes
        for param_idx, param in enumerate(self.network.parameters()):
            self.grad_aggregate_buffer.append(np.zeros(param.size()))

        self.network.to(torch.device("cpu"))

    def update_model(self):
        """Updates the model."""
        # gradients received from the workers are averaged used to update the model
        self.grad_aggregate_buffer = [
            i / (self.world_size - 1) for i in self.grad_aggregate_buffer
        ]
        for id, param in enumerate(self.network.parameters()):
            param.grad = self.grad_aggregate_buffer[id]

        self.optimizer.step()

    def aggregate_gradients(self, layer_idx, gradient):
        """Aggregates gradients from workers.

        Args:
            layer_idx: the index of the layer
            gradient: the gradient received from workers
        """
        self.grad_aggregate_buffer[layer_idx] = reduce(
            (lambda x, y: x + y), gradient[1:]
        )

    def recv_gradients(self):
        """Receives gradients from workers."""

        for layer_idx, layer in enumerate(self.network.parameters()):
            dummpy_grad = self.grad_accumulator.gradient_aggregator[layer_idx][0]
            dist.gather(
                dummpy_grad, self.grad_accumulator.gradient_aggregator[layer_idx], dst=0
            )
            self.aggregate_gradients(
                layer_idx=layer_idx,
                gradient=self.grad_accumulator.gradient_aggregator[layer_idx],
            )

    def save_model(self, save_path):
        """Saves the model.

        Args:
            save_path: the path to save the model
        """
        with open(save_path, "wb") as f:
            torch.save(self.network.state_dict(), f)
        return

    def broadcast_gradients(self):
        """Broadcasts gradients to workers."""
        for layer_idx, layer in enumerate(self.network.parameters()):
            layer_weight = layer.detach()
            dist.broadcast(layer_weight, src=0)

    def start(self):
        """Starts the training process."""
        for i in range(1, self.max_num_step + 1):
            print("Server is at step: {}".format(i))
            self.network.train()
            self.broadcast_gradients()
            self.recv_gradients()
            self.update_model()
            self.cur_step += 1
