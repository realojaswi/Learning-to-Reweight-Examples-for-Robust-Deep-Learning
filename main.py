import torch
import time
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import argparse

from dataset import CustomDataset
from model import SimpleCNN
from trainer import trainer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='')
parser.add_argument('--noise_ratio', default=0, type=float)
parser.add_argument('--class_ratio', default=100, type=float)
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the required size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

# Define the path to the folder containing the data
# folder = 'OriginalVAR'
folder = 'NewVAR'

data_path = f'./{folder}/train'
dataset = datasets.ImageFolder(root=data_path, transform=transform)

num_val_samples = int(0.2 * len(dataset))
num_train_samples = len(dataset) - num_val_samples

# Split the dataset into training and validation sets
train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_val_samples])

batch_size = 64
epochs = 100

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

val_dataset = datasets.ImageFolder(root=f'./{folder}/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print(f"train: size={len(train_loader)} classes={dataset.classes}")
print(f"test: size={len(test_loader)} classes={dataset.classes}")
print(f"val: size={len(val_loader)} classes={val_dataset.classes}")


# Make our dataset which can add biases to the dataset
def stack_data(dataloader):
    arr = [[], []]
    for sample in dataloader:
        arr[0].append(sample[0])
        arr[1].append(sample[1])
    arr[0] = torch.vstack(arr[0])
    arr[1] = torch.hstack(arr[1])
    return arr


train_data, test_data, val_data = stack_data(train_loader), stack_data(test_loader), stack_data(val_loader)
nb_class_ratios = {0: 1, 1: 1}
nb_noise_ratios = {0: 0, 1: 0}

class_ratios = {0: args.class_ratio/100, 1: 1}
noise_ratios = {0: args.noise_ratio/100, 1: 0}

title=f'Noise: {noise_ratios[0]*100:.0f}%, Ratio of two categories {max(class_ratios[0], class_ratios[1])/min(class_ratios[0], class_ratios[1]):.0f}:{1}'
print(title)

train_loader = DataLoader(CustomDataset(train_data[0], train_data[1], class_ratios, noise_ratios),
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(CustomDataset(test_data[0], test_data[1], nb_class_ratios, nb_noise_ratios),
                         batch_size=batch_size, shuffle=True)

# Collect 100 samples of the selected classes from test dataset as validation data
X_validation = val_data[0]
y_validation = val_data[1]
print(X_validation.shape)
print(y_validation.shape)

# Instantiate and train model using traditional method
model = SimpleCNN(n_out=1)
my_trainer_normal = trainer(model, train_loader, test_loader, X_validation, y_validation, device)
normal_acc = my_trainer_normal.train_normal(epochs=epochs)

# Instantiate and train model using proposed method
model2 = SimpleCNN(n_out=1)
my_trainer2 = trainer(model2, train_loader, test_loader, X_validation, y_validation, device)
reweighted_acc = my_trainer2.train_reweighted(epochs=epochs)

fig, ax = plt.subplots(1)
ax.plot(normal_acc, label='Normal Training')
ax.plot(reweighted_acc, label='Reweighted Training')
ax.set_xlabel('Epochs')
# ax.set_xticks(np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]))
# ax.set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax.set_ylabel('Accuracy')
plt.title(title)
plt.legend()
plt.savefig(f'{title}')

with open('results.txt', 'a') as f:
    f.write(f"{title} normal acc {np.mean(normal_acc):2f} {normal_acc[-1]:2f} reweight acc {np.mean(reweighted_acc):2f} {reweighted_acc[-1]:2f} \n")

