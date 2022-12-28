import torch
import torchvision
from .task import Task


class LabelPermutedMNIST(Task):
    """
    Iteratable MNIST task with permuted labels.
    Each sample is a 28x28 image and the label is a number between 0 and 9.
    The labels are permuted every 100 steps.
    """

    def __init__(self, name="label_permuted_mnist", batch_size=32, change_freq=100):
        self.dataset = self.get_dataset(False)
        self.change_freq = change_freq
        self.step = 0
        self.criterion = "cross_entropy"
        super().__init__(name, batch_size)

    def __next__(self):
        self.step += 1
        if self.step % self.change_freq == 0:
            self.change_all_lables()

        try:
            # Samples from dataset
            return next(self.iterator)
        except StopIteration:
            # restart the iterator if the previous iterator is exhausted.
            self.iterator = self.generator()
            return next(self.iterator)

    def generator(self):
        return iter(self.get_dataloader(self.dataset))

    def get_dataset(self, train=True):
        return torchvision.datasets.MNIST(
            "dataset",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
                ]
            ),
        )

    def get_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def change_all_lables(self):
        self.dataset.targets = (self.dataset.targets + 1) % 10
        self.iterator = iter(self.get_dataloader(self.dataset))


if __name__ == "__main__":
    task = LabelPermutedMNIST()
    for i, (x, y) in enumerate(task):
        print(x.shape, y.shape)
        if i == 10:
            break