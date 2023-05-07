from nn_train import NN_Trainer
import torch.distributed as dist
import torch
import numpy as np


class ModelBuffer(object):
    """This class is used to receive model weights from the parameter server

    Attributes:
        recv_buf: the buffer used to receive model weights from the parameter server
    """

    def __init__(self, network):
        """Initializes the instance with basic settings.

        Args:
            network: the network model
        """
        super(ModelBuffer, self).__init__()
        self.recv_buf = []
        # initialize the buffer
        for param_idx, param in enumerate(network.parameters()):
            self.recv_buf.append(torch.zeros(param.size()))


class Worker(NN_Trainer):
    """This class is used to train the model on the worker side

    Attributes:
        cur_step: the current training step
        next_step: the next training step
        rank: the rank of the worker
        batch_size: the batch size used for training
        max_epochs: the maximum number of epochs
        momentum: the momentum used for SGD
        lr: the learning rate used for SGD
        max_steps: the maximum number of training steps
        eval_freq: the frequency of evaluation
        train_dir: the directory used to store the training results
        device: the device used for training
        world_size: the number of workers
    """

    def __init__(self, **kwargs):
        """Initializes the instance with basic settings.

        Args:
            kwargs: the keyword arguments
        """
        super(NN_Trainer, self).__init__()

        self.cur_step = 0
        self.next_step = 0  # we will fetch this one from parameter server

        self.rank = kwargs["rank"]
        self.batch_size = kwargs["batch_size"]
        self.max_epochs = kwargs["max_epochs"]
        self.momentum = kwargs["momentum"]
        self.lr = kwargs["lr"]
        self.max_steps = kwargs["max_num_step"]
        self.eval_freq = kwargs["eval_freq"]
        self.train_dir = kwargs["train_dir"]
        self.device = kwargs["device"]
        self.world_size = kwargs["world_size"]

    def build_model(self, num_classes=10):
        """Builds the model.

        Args:
            num_classes: the number of classes. Defaults to 10.
        """
        super().build_model(num_classes)
        # assign a buffer for receiving models from parameter server
        self.model_recv_buf = ModelBuffer(self.network)
        self.network.to(self.device)

    def train(self, train_loader, test_loader):
        """Trains the model.

        Args:
            train_loader: the training data loader
            test_loader: the testing data loader
        """
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure the value we fetched from parameter server is 1
        print("Worker {}: starting training".format(self.rank))
        # start the training process
        for num_epoch in range(self.max_epochs):
            for batch_idx, (train_image_batch, train_label_batch) in enumerate(
                train_loader
            ):
                # worker exits
                if self.cur_step == self.max_steps:
                    break
                X_batch, y_batch = train_image_batch.to(
                    self.device
                ), train_label_batch.to(self.device)
                # receive weights from the server and update the model
                self.receive_gradients()
                # initialize the training
                self.network.train()
                self.optimizer.zero_grad()
                # forward
                logits = self.network(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()

                precision = self.accuracy(logits.detach(), train_label_batch.long())
                self.send_gradients()

                print(
                    "Worker: {}, Step: {}, Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Acc: {:.4f}".format(
                        self.rank,
                        self.cur_step,
                        num_epoch,
                        batch_idx * self.batch_size,
                        len(train_loader.dataset),
                        (
                            100.0
                            * (batch_idx * self.batch_size)
                            / len(train_loader.dataset)
                        ),
                        loss.item(),
                        np.array(precision)[0],
                    )
                )

                if self.cur_step % self.eval_freq == 0:
                    self.save_model(
                        file_path=self.train_dir + "step_" + str(self.cur_step)
                    )

    def receive_gradients(self):
        """receive gradients from parameter server"""
        for layer_idx, layer in enumerate(self.model_recv_buf.recv_buf):
            # receiver the tensor
            dist.broadcast(self.model_recv_buf.recv_buf[layer_idx], src=0)
        self.update_model(self.model_recv_buf.recv_buf)
        # Note that at here we update the global step
        self.cur_step += 1

    def update_model(self, weights_to_update):
        """write model fetched from parameter server to local model"""
        new_state_dict = {}
        model_index = 0
        for param_idx, (key_name, param) in enumerate(
            self.network.state_dict().items()
        ):
            # we do not update the `BatchNorm` layer
            if (
                "running_mean" in key_name
                or "running_var" in key_name
                or "num_batches_tracked" in key_name
            ):
                tmp_dict = {key_name: param}
            else:
                assert param.size() == weights_to_update[model_index].size()
                tmp_dict = {key_name: weights_to_update[model_index].to(self.device)}
                model_index += 1
            new_state_dict.update(tmp_dict)
        # loads the saved state dictionary into the model
        self.network.load_state_dict(new_state_dict)

    def send_gradients(self):
        """send gradients to parameter server"""
        for p_index, param in enumerate(self.network.parameters()):
            if self.device.type == "cuda":
                grad = param.grad.to(torch.device("cpu")).detach()
            else:
                grad = param.grad.detach()
            dist.gather(grad, [], dst=0)

    def save_model(self, file_path):
        """save the model to the file path"""
        torch.save(self.network)
