import torch
from torch.autograd import Variable
from model_ops.resnet import *


class NN_Trainer(object):
    """Class representing a neural network trainer.

    Attributes:
        batch_size: An integer indicating the batch size.
        max_epochs: An integer indicating the maximum number of epochs.
        lr: A float indicating the learning rate.
        momentum: A float indicating the momentum.
        network: A neural network model.
        optimizer: A torch optimizer.
        criterion: A loss function.
        cur_step: An integer indicating the current step.
        eval_batch_size: An integer indicating the batch size for evaluation.
        _epoch_counter: An integer indicating the current epoch.
    """

    def __init__(self, **kwargs):
        """Initializes the instance with basic settings.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        self.batch_size = kwargs["batch_size"]
        self.max_epochs = kwargs["max_epochs"]
        self.lr = kwargs["lr"]
        self.momentum = kwargs["momentum"]

    def build_model(self, num_classes=10):
        """Builds the neural network model.

        Args:
            num_classes: An integer indicating the number of classes. Defaults to 10.
        """
        self.network = ResNet18(num_classes)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(), lr=self.lr, momentum=self.momentum
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def accuracy(self, output, target, topK=(1,)):
        """Computes the precision@k for the specified values of k.

        Args:
            output: A torch tensor containing the output of the model.
            target: A torch tensor containing the target labels.
            topK: A tuple containing the top k values to compute precision. Defaults to (1,).
        """
        maxk = max(topK)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topK:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def evaluate(self, val_loader):
        """Evaluates the model.

        Args:
            val_loader: A torch DataLoader object containing the validation data.
        """
        self.network.eval()
        prec1_counter = batch_counter = 0
        while val_loader.dataset.epochs_completed <= self._epoch_counter:
            eval_image_batch, eval_label_batch = val_loader.next_batch(
                batch_size=self.eval_batch_size
            )
            X_batch, y_batch = Variable(eval_image_batch.float()), Variable(
                eval_label_batch.long()
            )
            output = self.network(X_batch)
            prec1_tmp = self.accuracy(output.detach(), eval_label_batch.long())
            prec1_counter += prec1_tmp
            batch_counter += 1
        prec1 = prec1_counter / batch_counter
        self._epoch_counter = val_loader.dataset.epochs_completed
        print(
            "Evaluation Performance: Step:{} Precision@1: {}".format(
                self.cur_step, prec1.numpy()[0]
            )
        )
