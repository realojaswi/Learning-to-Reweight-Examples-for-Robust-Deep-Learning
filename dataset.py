from matplotlib import pyplot as plt
from torchvision import datasets
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class CustomDataset(Dataset):
    def __init__(self, X, y, class_ratios, noise_ratios):
        self.X = X
        self.y = y
        if type(self.y) == torch.Tensor:
            self.y = self.y.numpy()
        if type(self.X) == torch.Tensor:
            self.X = self.X.numpy().transpose([0, 2, 3, 1])
        N = len(X)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        #  get unique labels and their count in the dataset
        self.unique_y, self.y_counts = np.unique(self.y, return_counts=True)
        assert len(class_ratios) == len(self.unique_y)
        assert len(noise_ratios) == len(self.unique_y)
        print(f"Unique labels {self.unique_y}, Count {self.y_counts}")

        # repeat unbalanced data for accurate metrics (should we do this?)
        max_count = max(self.y_counts)
        _X = []
        _y = []
        for label in self.unique_y:
            idx = np.where(self.y == label)[0]
            _X.append(np.repeat(self.X[idx], max_count//len(idx), axis=0))
            _y.append(np.repeat(self.y[idx], max_count//len(idx), axis=0))
        self.X = np.vstack(_X)
        self.y = np.hstack(_y)

        # normalize class ratio list
        self.class_ratios = class_ratios
        # ratio_sum = sum(self.class_ratios.values())
        # for key in class_ratios.keys():
        #     self.class_ratios[key] /= ratio_sum

        # prune dataset
        for label in self.unique_y:
            idx = np.where(self.y == label)[0]
            n_samples = min(len(idx), int(self.class_ratios[label] * len(idx)))
            to_remove = np.random.choice(idx, len(idx) - n_samples, replace=False)
            self.y[to_remove] = -10000
        self.X = self.X[self.y != -10000]
        self.y = self.y[self.y != -10000]

        self.unique_y, self.y_counts = np.unique(self.y, return_counts=True)

        print(f"Unique labels {self.unique_y}, Count {self.y_counts} (After pruning)")

        # add noise to each class
        self.noise_ratios = noise_ratios
        for label in self.unique_y:
            idx = np.where(self.y == label)[0]
            n_noise = min(len(idx), int(self.noise_ratios[label] * len(idx)))
            to_noise = np.random.choice(idx, n_noise, replace=False)
            self.y[to_noise] = np.random.choice(self.unique_y, n_noise, replace=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image, label = self.X[idx], self.y[idx]
        # image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        image = self.transform(image)
        return image, label


if __name__ == "__main__":
    batch_size = 64

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    labels = mnist_train.targets.numpy()
    data = mnist_train.data.numpy()
    class_ratios = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 1, 9: 0}
    noise_ratios = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0.1, 5: 0, 6: 0, 7: 0, 8: 0.1, 9: 0}
    # noise ratio should be < 1 for each class. If no noise 0

    train_loader = DataLoader(CustomDataset(data, labels, class_ratios, noise_ratios), batch_size=batch_size,
                              shuffle=True)

    """Runner code"""
    for sample in train_loader:
        break
    print(sample[0].shape)
    plt.imshow(sample[0][0].numpy().transpose([1, 2, 0]))
    plt.title(int(sample[1][0]))
    plt.show()
