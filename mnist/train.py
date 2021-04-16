import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

test_dataset = datasets.MNIST('../data/MNIST', train = False, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)