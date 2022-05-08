from HVAE import HVAE
import torch
import numpy as np
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

# Get dataset 
argparser = argparse.ArgumentParser()
argparser.add_argument('-data',type=str,default="MNIST")
args = argparser.parse_args()

print(args.data)

# Determine correct save path 
if args.data == "MNIST":
    path = '/zhome/71/2/146488/bachelor_project/FID_score/MNIST_IWAE'
    batches = 40
    images = 10240
    sl = 3
if args.data == "OMNIGLOT":
    path = '/zhome/71/2/146488/bachelor_project/FID_score/OMNIGLOT_IWAE'
    batches = 32
    images = 8192
    sl = 4

# Initialize model with trained weights
model = HVAE(in_features=784, layers=sl, batch_norm=0, device=device).to(device)   

# Initialize weights with Glorot optimization (as in LVAE paper)
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

model.apply(init_weights)

# Load model 
if args.data == "MNIST":
    model.load_state_dict(torch.load('saved_models/model_MNIST_IWAE.pth'))

if args.data == "OMNIGLOT":
    model.load_state_dict(torch.load('saved_models/model_OMNIGLOT_IWAE.pth'))
    
model.eval()

# Save images 
im_count = 1

for i in range(batches):
    # Generate new images 
    px = model.sample_from_prior(batch_size=256)['PxGz1']
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
        
  
        
    