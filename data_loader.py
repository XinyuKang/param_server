import os
import torch
from torchvision import datasets, transforms
from filelock import FileLock


class MNISTEvenOddDataset(torch.utils.data.Dataset):
    """MNIST dataset with even/odd labels.

    Args:
        img_data: a dataset
        labels: a list of labels
    """

    def __init__(self, ready_data):
        """Initializes the dataset.

        Args:
            ready_data: a dataset
        """
        self.img_data = ready_data.data
        self.labels = ready_data.targets % 2

    def __len__(self):
        """Gets the length of the dataset.

        Returns:
            the length of the dataset
        """
        return len(self.labels)

    def __getitem__(self, ind):
        """Gets an item from the dataset.

        Args:
            ind: an index

        Returns:
            an item from the dataset
        """
        return torch.true_divide(
            self.img_data[ind].view(-1, 28 * 28).squeeze(), 255
        ), torch.tensor([self.labels[ind]])


def get_data_loader():
    """Safely downloads data. Returns training/validation set dataloader.

    Returns:
        train_loader: training set dataloader
        test_loader: validation set dataloader
    """
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.

    with FileLock(os.path.expanduser("~/data.lock")):
        train_dataset = datasets.MNIST(
            "~/data", train=True, download=True, transform=mnist_transforms
        )

        test_dataset = datasets.MNIST("~/data", train=False, transform=mnist_transforms)

        train_loader = torch.utils.data.DataLoader(
            MNISTEvenOddDataset(train_dataset),
            batch_size=128,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            MNISTEvenOddDataset(test_dataset),
            batch_size=128,
            shuffle=False,
        )
    return train_loader, test_loader
