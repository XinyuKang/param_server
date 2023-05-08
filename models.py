import torch
import torch.nn as nn


class MLP(nn.Module):
    """Small Linear Network for MNIST.

    Attributes:
        fc_weights: a list of weights
        fc_bias: a bias
    """

    def __init__(self):
        """Initializes a linear network."""
        super(MLP, self).__init__()
        self.fc_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(1)) for weight in range(784)]
        )
        init_fc = [nn.init.uniform_(x) for x in self.fc_weights]

        self.fc_bias = nn.Parameter(torch.empty(1))
        nn.init.uniform_(self.fc_bias)

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x: an input tensor

        Returns:
            an output tensor
        """
        for i, param in enumerate(self.fc_weights):
            if i == 0:
                p = x[:, i] * param
            else:
                p += x[:, i] * param
        x = p.unsqueeze(1) + self.fc_bias
        return x

    def get_weights(self):
        """Gets the weights of the network.

        Returns:
            a list of weights
        """

        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, keys, weights):
        """Sets the weights of the network.

        Args:
            keys: a list of keys
            weights: a list of weights
        """
        flatten_weights = [item for sublist in weights for item in sublist]
        self.load_state_dict({keys[i]: flatten_weights[i] for i in range(len(keys))})

    def get_gradients(self, keys):
        """Gets the gradients of the network.

        Args:
            keys: a list of keys
        """
        grads = {}

        for name, p in self.named_parameters():
            if name in keys:
                grad = None if p.grad is None else p.grad.data.cpu().numpy()
                grads[name] = grad

        return [grads[key] for key in keys]

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset.

    Args:
        test_loader: a data loader for the validation dataset

    Returns:
        accuracy: the accuracy of the model on the validation dataset
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # This is only set to finish evaluation faster.
            if batch_idx * len(data) > 1024:
                break
            outputs = nn.Sigmoid()(model(data))
            predicted = outputs > 0.5
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total
