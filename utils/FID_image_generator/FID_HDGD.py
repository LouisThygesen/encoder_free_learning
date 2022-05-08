
from HDGD import HDGD

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import DataLoader, Dataset

import numpy as np
from scipy import io
import random
import argparse
import matplotlib.pyplot as plt

# Seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Set up GPU support
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using GPU!')
else:
    device = torch.device("cpu")
    print('Using CPU!')
    
# Get dataset 
argparser = argparse.ArgumentParser()
argparser.add_argument('-data',type=str,default="MNIST")
args = argparser.parse_args()

print(args.data)

# Determine correct save path 
if args.data == "MNIST":
    path = '/zhome/71/2/146488/bachelor_project/FID_score/MNIST_HDGD'
    batches = 40
    images = 10240
if args.data == "OMNIGLOT":
    path = '/zhome/71/2/146488/bachelor_project/FID_score/OMNIGLOT_HDGD'
    batches = 32
    images = 8192
    
# Function for loading data
def load_data():
    if args.data == "MNIST":
        # Transform: Flatten image and use pixels for Bernoulli (image is sample)
        transform = Compose([
            ToTensor(),
            Lambda(lambda x: torch.flatten(x)),
            Lambda(lambda x: torch.bernoulli(x)),
        ])

        # Define MNIST custom dataset
        class mnist(Dataset):
            def __init__(self, mode):
                if mode == "train":
                    self.data = MNIST("../data", train=True, transform=transform, download=True)
                if mode == "test":
                    self.data = MNIST("../data", train=False, transform=transform, download=True)

            def __getitem__(self, index):
                data, target = self.data[index]

                return data, target, index

            def __len__(self):
                return len(self.data)

        data_train = mnist(mode="train")
        data_test = mnist(mode="test")

    if args.data == "OMNIGLOT":
        # Dataset class
        class omniglot(Dataset):
            def __init__(self, mode):
                mat = io.loadmat('../data/omniglot28x28.mat')

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

                # Transform
                self.transform = lambda x: torch.bernoulli(x)

            def __getitem__(self, idx):
                # Images are not sampled from Bernoulli distribution (used directly)
                images = self.transform(self.images[idx, :])
                labels = self.labels[idx]

                return images, labels, idx

            def __len__(self):
                return self.images.size(0)

        # Load train and test datasets
        data_train = omniglot(mode="train")
        data_test = omniglot(mode="test")

    # Create data loaders
    train_loader = DataLoader(data_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=32, shuffle=True)

    return train_loader, test_loader

# Load and pre-process data
train_loader, test_loader = load_data()

# Load model
train_size = len(train_loader.dataset)
test_size = len(train_loader.dataset) # TODO:

# Load model
class contemp_config():
    def __init__(self):
        self.lat_dim = 64
        self.theta_prior = 0
        self.sl = 5
        self.bn = 1
        
config = contemp_config()
model = HDGD(config, train_size, test_size).to(device)

# Load model 
if args.data == "MNIST":
    model.load_state_dict(torch.load('saved_models/model_MNIST_HDGD.pth'))

if args.data == "OMNIGLOT":
    model.load_state_dict(torch.load('saved_models/model_OMNIGLOT_HDGD.pth'))
    
model.eval()

# Save images 
im_count = 1

for i in range(batches):
    # Generate new images 
    px = model.sample_from_prior(batch_size=256)['PxGz']
    x = px.sample()
    
    for j in range(x.size(0)):
        # Get image from batch 
        image = x[j,:]
        
        # Update save path 
        im_path = f"{path}/img{im_count}"
        
        # Plot and save 
        plt.figure()
        plt.imshow(image.view(28,28).detach().to('cpu'), cmap='gray')
        plt.axis('off')
        plt.savefig(im_path, bbox_inches='tight', pad_inches = 0)
        plt.close()
        plt.cla()
        plt.clf()
        
        im_count += 1

    # Show progress
    print(f'Progress: {(i+1)/batches * 100}%')
    
    
    
    
    
    
    