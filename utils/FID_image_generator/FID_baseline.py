

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import DataLoader, Dataset

import torch
import scipy.io
import numpy as np
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using GPU!')
else:
    device = torch.device("cpu")
    print('Using CPU!')

def load_data(dataset):
    """ Description: Load dataset and create data loaders
        Return: train_loader, test_loader
    """

    if dataset == "MNIST":
        # Transform function
        transform = Compose([
            ToTensor(),
            Lambda(lambda x: torch.flatten(x)),
            Lambda(lambda x: torch.bernoulli(x)),
        ])

        # Load train and test datasets
        data_train = MNIST("../data", train=True, transform=transform, download=True)
        data_test = MNIST("../data", train=False, transform=transform, download=True)

    if dataset == "OMNIGLOT":
        # Dataset class
        class omniglot(Dataset):
            def __init__(self, mode):
                mat = scipy.io.loadmat('../data/omniglot28x28.mat')

                # Get images
                if mode == "train":
                    raw_images = np.transpose(mat['data'])
                if mode == "test":
                    raw_images = np.transpose(mat['testdata'])

                self.images = torch.from_numpy(raw_images).float()

                # Get labels
                if mode == "train":
                    raw_labels = np.argmax(mat['target'], 0) + 1
                if mode == "test":
                    raw_labels = np.argmax(mat['testtarget'], 0) + 1

                self.labels = torch.from_numpy(raw_labels).float()

                # Define transform
                self.transform = lambda x: torch.bernoulli(x)

            def __getitem__(self, idx):
                # Images are sampled from Bernoulli distribution
                # images = self.transform(self.images[idx, :])
                images = self.transform(self.images[idx, :])
                labels = self.labels[idx]

                return images, labels

            def __len__(self):
                return self.images.size(0)

        # Load train and test datasets
        data_train = omniglot(mode="train")
        data_test = omniglot(mode="test")

    # Create data loaders
    train_loader = DataLoader(data_train, 256, shuffle=True)
    test_loader = DataLoader(data_test, 256, shuffle=True)

    return train_loader, test_loader

# Chose dataset from console
argparser = argparse.ArgumentParser()
argparser.add_argument('-data',type=str,default="OMNIGLOT")
args = argparser.parse_args()

print(args.data)

# Determine correct save path 
if args.data == "MNIST":
    path = '/zhome/71/2/146488/bachelor_project/FID_score/MNIST'
if args.data == "OMNIGLOT":
    path = '/zhome/71/2/146488/bachelor_project/FID_score/OMNIGLOT'

# Load and pre-process data
_, test_loader = load_data(args.data)

# Save images 
im_count = 1

for x, y in test_loader:
    for i in range(x.size(0)):
        # Get image from batch 
        image = x[i,:]
        
        # Update save path 
        im_path = f"{path}/img{im_count}"
        
        # Plot and save 
        plt.figure()
        plt.imshow(image.view(28,28), cmap='gray')
        plt.axis('off')
        plt.savefig(im_path, bbox_inches='tight', pad_inches = 0)
        plt.close()
        plt.cla()
        plt.clf()
        
        # Progress tracker 
        im_count += 1
        
        if im_count%200 == 0:
            print(f'Progress: {int(im_count/len(test_loader.dataset) * 100)}%')
        
        
    
    
    
    
    
    





