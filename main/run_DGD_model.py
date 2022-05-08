# FINISHED: DO NOT CHANGE ANYTHING

from DGD import DGD
from DGD_GMM import DGD_GMM
from DGD_CONV import DGD_CONV

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import DataLoader, Dataset

import numpy as np
from scipy import io
import pandas as pd
from collections import defaultdict
import random

import os
import wandb
import argparse
from tqdm import tqdm
from sklearn import decomposition
import seaborn as sns
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

# Auto-login (wandb)
os.environ["WANDB_API_KEY"] = "44b27736c17cd4318936992ac992d9d787f1a5e5"

# Global data save (not great - but it works)
train_data = defaultdict(list)
val_data = defaultdict(list)

def hyperparameters():
    """ Description: Gets hyper-parameters from command line and stores them in an object
        Return: config-object """

    # Get varying hyper-parameters from command line
    argparser = argparse.ArgumentParser(description='Get hyper-parameters')
    argparser.add_argument('-data',type=str,default="MNIST")
    argparser.add_argument('-model',type=str,default="DGD")
    argparser.add_argument('-epochs',type=int,default=200)
    argparser.add_argument('-opt_mlp', type=str, default="ADAM")
    argparser.add_argument('-opt_zs', type=str, default="ADAM")
    argparser.add_argument('-lr_mlp', type=float, default=1e-3)
    argparser.add_argument('-lr_zs', type=float, default=1e-2)
    argparser.add_argument('-wd_mlp', type=float, default=1e-5)
    argparser.add_argument('-lat_dim', type=int, default=32)
    argparser.add_argument('-theta_prior', type=int, default=0)

    args = argparser.parse_args()

    # Create hyper-parameter object
    class config():
        def __init__(self,data,model,epochs,opt_mlp,opt_zs,lr_mlp,lr_zs,wd_mlp,lat_dim,theta_prior):
            self.data = data
            self.model = model
            self.epochs = epochs
            self.train_bs = 32
            self.test_bs = 32
            self.opt_mlp = opt_mlp
            self.opt_zs = opt_zs
            self.lr_mlp = lr_mlp
            self.lr_zs = lr_zs
            self.wd_mlp = wd_mlp
            self.lat_dim = lat_dim
            self.main_path = ""
            self.opt_steps = 50
            self.theta_prior = theta_prior

    return config(**vars(args))

def load_data(config):
    if config.data == "MNIST":
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

    if config.data == "OMNIGLOT":
        # Define OMNIGLOT custom dataset
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
                # Images are sampled from Bernoulli distribution
                images = self.transform(self.images[idx, :])
                labels = self.labels[idx]

                return images, labels, idx

            def __len__(self):
                return self.images.size(0)

        # Load train and test datasets
        data_train = omniglot(mode="train")
        data_test = omniglot(mode="test")

    # Create data loaders
    train_loader = DataLoader(data_train, batch_size=config.train_bs, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=config.test_bs, shuffle=True)

    return train_loader, test_loader

def load_model(config, train_size, test_size):
    if config.model == "DGD":
        model = DGD(config, train_size, test_size).to(device)
    if config.model == "DGD_GMM":
        model = DGD_GMM(config, train_size, test_size).to(device)
    if config.model == "DGD_CONV":
        model = DGD_CONV(config, train_size, test_size).to(device)
        
    if config.theta_prior:
        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight)

        model.apply(init_weights)

    return model

def build_optimizers(config, model):
    # Store all train optimizers in lists
    train_opts = []

    # Set up network optimizer
    if config.opt_mlp == "ADAM":
        opt_mlp = torch.optim.Adam(params=model.mlp.parameters(), lr=config.lr_mlp,
                                   weight_decay=config.wd_mlp)

    if config.opt_mlp == "SGD":
        opt_mlp = torch.optim.SGD(params=model.mlp.parameters(), momentum=0.9, lr=config.lr_mlp,
                                 weight_decay=config.wd_mlp)

    train_opts.append(opt_mlp)

    # Set up latent representation optimizer
    if config.opt_zs == "ADAM":
        opt_zs_train = torch.optim.Adam(params=[model.zs_train], lr=config.lr_zs)
        opt_zs_test = torch.optim.Adam(params=[model.zs_test], lr=config.lr_zs)

    if config.opt_zs == "SGD":
        opt_zs_train = torch.optim.SGD(params=[model.zs_train], momentum=0.9, lr=config.lr_zs)
        opt_zs_test = torch.optim.SGD(params=[model.zs_test], momentum=0.9, lr=config.lr_zs)

    train_opts.append(opt_zs_train)

    # Set up \phi optimizer if necessary
    if (config.model == "DGD_GMM") or (config.model == "DGD_CONV"):
        opt_phi = torch.optim.Adam(params=[model.mixture_coeffs, model.Gaussian_mean, model.Gaussian_other], lr=0.1)

        train_opts.append(opt_phi)

    return train_opts, opt_zs_test

def train_epoch(config, model, train_loader, train_opts, epoch):
    # Store batch results where keys are performance stats names and values are list of batch results
    training_batch_data = defaultdict(list)

    with tqdm(train_loader, unit="batch", desc=f" {epoch+1}/{config.epochs}") as batch:
        for (x, y, idx) in batch:
            x = x.to(device)

            # Use zero_grad on all optimizers
            for opt in train_opts:
                opt.zero_grad()

            # Compute loss from forward pass
            train_loss, diagnostics = model.loss(x, idx=idx, mode="train")

            # Take optimization step (first network, then zs and at last \phi)
            train_loss.backward()

            for opt in train_opts:
                opt.step()

            # Collect mean diagnostic data for each batch
            for k, v in diagnostics.items():
                training_batch_data[k] += [v.mean().item()]

        # Mean diagnostic data for finished epoch (by taking mean of batch means)
        for k, v in training_batch_data.items():
            train_data[k] += [np.mean(training_batch_data[k])]

def test_epoch(config, model, train_loader, test_loader, opt_zs_test, epoch):
    validation_batch_data = defaultdict(list)

    # Best loss up till now
    if epoch == 0:
        best_loss = 5000
    else:
        best_loss = min(val_data['loss'][:-4])

    # Optimize test zs
    model.train()

    for opt_step in range(config.opt_steps):
        print(f"Optimization epoch: {opt_step+1}/{config.opt_steps}")

        for (x, y, idx) in test_loader:
            x = x.to(device)

            # Use zero_grad on optimizer
            opt_zs_test.zero_grad()

            # Compute loss from forward pass
            train_loss, diagnostics = model.loss(x, idx=idx, mode="test")

            # Take optimization step (first network, then zs and at last \phi)
            train_loss.backward()
            opt_zs_test.step()

    # Evaluate model
    model.eval()

    with torch.no_grad():
        for (x, y, idx) in test_loader:
            x = x.to(device)

            # Compute loss from forward pass
            val_loss, diagnostics = model.loss(x, idx=idx, mode="test")

            # Collect mean diagnostic data for each batch
            for k, v in diagnostics.items():
                validation_batch_data[k] += [v.mean().item()]

        # Mean diagnostic data for evaluation epoch (by taking mean of batch means)
        for k, v in validation_batch_data.items():
            val_data[k] += [np.mean(validation_batch_data[k])]

        # Prototyping plots 
        if epoch%20 == 0: 
            plot_wandb(model, train_loader, test_loader)
            
        # Check every 20th epoch if the model has become better. If so then save the model. Only do this if the script is
        # run as main script (e.g. don't do it during hyper-parameter search)
        if (val_data['loss'][-1] < best_loss):
            # Save model
            torch.save(model.state_dict(), f"{config.main_path}/model.pth")
    
            # Save number of best epoch
            lol = pd.Series()
            lol['best_epoch'] = epoch + 1
            lol.to_csv(f'{config.main_path}/best_epoch.csv')
        
def plot_wandb(model, train_loader, test_loader):
    model.eval()
    
    """ Plot 1: Images generated from prior """
    # Get data example (to determine batch size)
    x, y, idx = next(iter(train_loader))
    x = x.to(device)

    # Get p(x|z) for z generated from prior p(z) and sample from it
    px = model.sample_from_prior(batch_size=x.size(0))['PxGz']
    x_samples = px.mean

    # Plot images originating from 5 of the prior samples (wandb)
    images = [wandb.Image(x_samples[i, :].view(28, 28)) for i in range(5)]
    images = pd.DataFrame({"": images})

    image_save1 = wandb.Table(data=images)

    """ Plot 2: Input and reconstruction (training data) """
    x, y, idx = next(iter(train_loader))
    x = x.to(device)

    # Get observation model from x->z-->x
    PxGz, _ = model(x, idx, mode="train")
    x_rec = PxGz.mean

    # Collect original images and reconstructions
    images_A = [wandb.Image(x[i, :].view(28, 28)) for i in range(5)]
    images_B = [wandb.Image(x_rec[i, :].view(28, 28)) for i in range(5)]

    # Save as wandb-table
    images = pd.DataFrame({"Label": y[:5],
                           "Original image": images_A,
                           "Reconstructed image": images_B
                           })

    image_save2 = wandb.Table(data=images)

    """ Plot 3: PCA latent representation plot (training data) """
    pca = decomposition.PCA(n_components=2)

    ys = []
    zs = 0
    count = 0

    for (x, y, idx) in train_loader:
        if count == 0:
            zs = model.zs_train[idx].detach().cpu()
            count += 1

        else:
            lol = model.zs_train[idx].detach().cpu()
            zs = torch.vstack((zs, lol))

        y = y.tolist()
        ys += [str(element) for element in y]

    # PCA on zs
    pca.fit(zs)
    x = pca.transform(zs)

    colors = {'0': 'black', '1': 'red', '2': 'green', '3': 'yellow', '4': 'brown', '5': 'cyan', '6': 'pink',
              '7': 'orange', '8': 'chocolate', '9': 'darkkhaki'}

    # Structure data and plot it
    df = pd.DataFrame(dict(x1=x[:, 0], x2=x[:, 1], y=ys))

    fig, ax = plt.subplots()

    if config.data == "OMNIGLOT":
        sns.scatterplot(x='x1', y='x2', data=df, hue='y', size=3, alpha=0.1)
    else:
        colors = {'0': 'black', '1': 'red', '2': 'green', '3': 'yellow', '4': 'brown', '5': 'cyan', '6': 'pink',
                  '7': 'orange', '8': 'chocolate', '9': 'darkkhaki'}
        sns.scatterplot(x='x1', y='x2', data=df, hue='y', palette=colors, size=3, alpha=0.1)

    # Save to wandb
    image_save3 = wandb.Image(plt)

    """ Plot 4: Input and reconstruction (testing data) """
    x, y, idx = next(iter(test_loader))
    x = x.to(device)

    # Get observation model from x->z-->x
    PxGz, _ = model(x, idx, mode="test")
    x_rec = PxGz.mean

    # Collect original images and reconstructions
    images_A = [wandb.Image(x[i, :].view(28, 28)) for i in range(5)]
    images_B = [wandb.Image(x_rec[i, :].view(28, 28)) for i in range(5)]

    # Save as wandb-table
    images = pd.DataFrame({"Label": y[:5],
                           "Original image": images_A,
                           "Reconstructed image": images_B
                           })

    image_save4 = wandb.Table(data=images)

    """ Plot 5: PCA latent representation plot (testing data) """
    ys = []
    zs = 0
    count = 0

    for (x, y, idx) in test_loader:

        if count == 0:
            zs = model.zs_test[idx].detach().cpu()

        else:
            lol = model.zs_test[idx].detach().cpu()
            zs = torch.vstack((zs, lol))

        y = y.tolist()
        ys += [str(element) for element in y]

        count += 1

    # PCA on zs
    x = pca.transform(zs)

    colors = {'0': 'black', '1': 'red', '2': 'green', '3': 'yellow', '4': 'brown', '5': 'cyan', '6': 'pink',
              '7': 'orange', '8': 'chocolate', '9': 'darkkhaki'}

    # Structure data and plot it
    df = pd.DataFrame(dict(x1=x[:, 0], x2=x[:, 1], y=ys))

    fig, ax = plt.subplots()
    if config.data == "OMNIGLOT":
        sns.scatterplot(x='x1', y='x2', data=df, hue='y', size=3, alpha=0.1, legend=False)
    else:
        colors = {'0': 'black', '1': 'red', '2': 'green', '3': 'yellow', '4': 'brown', '5': 'cyan', '6': 'pink',
                  '7': 'orange', '8': 'chocolate', '9': 'darkkhaki'}
        sns.scatterplot(x='x1', y='x2', data=df, hue='y', palette=colors, size=3, alpha=0.1)

    # Save to wandb
    image_save5 = wandb.Image(plt)

    """ Log plots to wandb """
    data = {
        "Images generated from prior": image_save1,
        "Original and reconstruction (training)": image_save2,
        "Latent representation (training)": image_save3,
        "Original and reconstruction (testing)": image_save4,
        "Latent representation (testing)": image_save5,
    }

    wandb.log(data)

def log_final_stats(config, model):   
    """ Part 1: Calculate test statistic """
    vars = torch.var(model.zs_test,axis=0) # TODO: Check om dimension er korrekt
    activated = torch.sum(vars > 10 ** (-2)).detach().cpu().item()

    # Save
    lol = pd.Series()
    lol['activation'] = activated
    lol.to_csv(f'{config.main_path}/activation_stats.csv')

    """ Part 2: Log loss statistics """
    # Use dataframe as main log
    main_log = pd.DataFrame()

    # Log training epoch results
    main_log['train_loss'] = np.array(train_data['loss'])
    main_log['train_log_PxGz'] = np.array(train_data['log_PxGz'])
    main_log['train_log_Pz'] = np.array(train_data['log_Pz'])

    # Log test epoch results
    main_log['val_loss'] = np.array(val_data['loss'])
    main_log['val_log_PxGz'] = np.array(val_data['log_PxGz'])
    main_log['val_log_Pz'] = np.array(val_data['log_Pz'])

    main_log.to_csv(f'{config.main_path}/main_log.csv')

def log_final_plots(config, model, train_loader, test_loader):
    """ Plot 1: Images generated from prior """
    # Get data example to determine batch size
    x, y, idx = next(iter(test_loader))
    x = x.to(device)

    # Get p(x|z) for z generated from prior p(z) and sample from it
    px = model.sample_from_prior(batch_size=x.size(0))['PxGz']
    x_samples = px.mean

    for i in range(30):
        x_sample = x_samples[i, :].view(28,28)

        # Create plot
        fig, ax = plt.subplots()
        ax.imshow(x_sample.detach().cpu(), cmap='gray')

        # Save plot
        save_path = f'{config.main_path}/priors'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        fig.savefig(f'{save_path}/prior{i}')
        plt.close()

    """ Plot 2: Input and reconstruction (training data) """
    x, y, idx = next(iter(train_loader))
    x = x.to(device)

    # Get observation model from x->z-->x
    PxGz, _ = model(x, idx, mode="train")
    x_rec = PxGz.mean

    for i in range(30):
        reconstruction = x_rec[i, :].view(28, 28)
        original = x[i, :].view(28, 28)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.imshow(original.cpu(), cmap='gray')
        ax2.imshow(reconstruction.detach().cpu(), cmap='gray')

        # Save
        save_path = f'{config.main_path}/train_recs'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        fig.savefig(f'{save_path}/rec{i}')
        plt.close()

    """ Plot 3: PCA latent representation plot (training data) """
    pca = decomposition.PCA(n_components=2)

    ys = []
    zs = 0
    count = 0

    for (x, y, idx) in train_loader:

        if count == 0:
            zs = model.zs_train[idx].detach().cpu()

        else:
            lol = model.zs_train[idx].detach().cpu()
            zs = torch.vstack((zs, lol))

        y = y.tolist()
        ys += [str(element) for element in y]

        count += 1

    # PCA on zs
    pca.fit(zs)
    x = pca.transform(zs)

    # Structure data and plot it
    df = pd.DataFrame(dict(x1=x[:, 0], x2=x[:, 1], y=ys))

    fig, ax = plt.subplots()

    if config.data == "OMNIGLOT":
        sns.scatterplot(x='x1', y='x2', data=df, hue='y', size=3, alpha=0.1, legend=False)
    else:
        colors = {'0': 'black', '1': 'red', '2': 'green', '3': 'yellow', '4': 'brown', '5': 'cyan', '6': 'pink',
                  '7': 'orange', '8': 'chocolate', '9': 'darkkhaki'}
        sns.scatterplot(x='x1', y='x2', data=df, hue='y', palette=colors, size=3, alpha=0.1)

    # Save plot
    save_path = f'{config.main_path}/train_lat_reps'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig.savefig(f'{save_path}/layer0')
    plt.close()

    """ Plot 4: Input and reconstruction (testing data) """
    x, y, idx = next(iter(test_loader))
    x = x.to(device)

    # Get observation model from x->z-->x
    PxGz, _ = model(x, idx, mode="test")
    x_rec = PxGz.mean

    for i in range(30):
        reconstruction = x_rec[i, :].view(28, 28)
        original = x[i, :].view(28, 28)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.imshow(original.cpu(), cmap='gray')
        ax2.imshow(reconstruction.detach().cpu(), cmap='gray')

        # Save
        save_path = f'{config.main_path}/test_recs'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        fig.savefig(f'{save_path}/rec{i}')
        plt.close()

    """ Plot 5: PCA latent representation plot (testing data) """
    ys = []
    zs = 0
    count = 0

    for (x, y, idx) in test_loader:

        if count == 0:
            zs = model.zs_test[idx].detach().cpu()
            count += 1

        else:
            lol = model.zs_test[idx].detach().cpu()
            zs = torch.vstack((zs, lol))

        y = y.tolist()
        ys += [str(element) for element in y]

    # PCA on zs
    x = pca.transform(zs)

    # Structure data and plot it
    df = pd.DataFrame(dict(x1=x[:, 0], x2=x[:, 1], y=ys))

    fig, ax = plt.subplots()

    if config.data == "OMNIGLOT":
        sns.scatterplot(x='x1', y='x2', data=df, hue='y', size=3, alpha=0.1, legend=False)
    else:
        colors = {'0': 'black', '1': 'red', '2': 'green', '3': 'yellow', '4': 'brown', '5': 'cyan', '6': 'pink',
                  '7': 'orange', '8': 'chocolate', '9': 'darkkhaki'}
        sns.scatterplot(x='x1', y='x2', data=df, hue='y', palette=colors, size=3, alpha=0.1)

    # Save plot
    save_path = f'{config.main_path}/test_lat_reps'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig.savefig(f'{save_path}/layer0')
    plt.close()

def train(config):
    # Set up weight and biases logging (fast-prototyping)
    wandb.init(config=config, project="bachelor_project", entity="louisdt")
    wandb.run.name = f"{config.data} - {config.model}"

    # Load and preprocess data
    train_loader, test_loader = load_data(config)

    # Load model
    train_size = len(train_loader.dataset)
    test_size = len(train_loader.dataset) # TODO

    model = load_model(config, train_size, test_size)

    # Load optimizers
    train_opts, opt_zs_test = build_optimizers(config, model)

    # Train and evaluate model
    for epoch in range(config.epochs):
        # Train on 1 epoch
        model.train()
        train_epoch(config, model, train_loader, train_opts, epoch)

        # Test on 1 epoch every 10th epoch - in the rest epoch just copy previous result (creates staircase diagram).
        # The model is evaluated so few times as the test optimization takes a lot of time.
        if epoch % 5 == 0:
            model.reset_zs_test() 
            opt_zs_test = torch.optim.Adam(params=[model.zs_test], lr=config.lr_zs) 
            test_epoch(config, model, train_loader, test_loader, opt_zs_test, epoch)
        else:
            val_data['loss'] += [val_data['loss'][-1]]
            val_data['log_PxGz'] += [val_data['log_PxGz'][-1]]
            val_data['log_Pz'] += [val_data['log_Pz'][-1]]
                
        # Log loss to wandb
        data = {
            'train_loss': train_data['loss'][-1],
            'train_log_PxGz': train_data['log_PxGz'][-1],
            'train_log_Pz': train_data['log_Pz'][-1],
            'val_loss': val_data['loss'][-1],
            'val_log_PxGz': val_data['log_PxGz'][-1],
            'val_log_Pz': val_data['log_Pz'][-1],
        }

        wandb.log(data)
            
    # Log for best model 
    model.load_state_dict(torch.load(f"{config.main_path}/model.pth"))

    # Train test latent representations
    model.train()

    for opt_step in range(config.opt_steps):
        print(f"Optimization epoch (results 1): {opt_step + 1}/{config.opt_steps}")

        for (x, y, idx) in test_loader:
            x = x.to(device)

            # Use zero_grad on optimizer
            opt_zs_test.zero_grad()

            # Compute loss from forward pass
            train_loss, diagnostics = model.loss(x, idx=idx, mode="test")

            # Take optimization step (first network, then zs and at last \phi)
            train_loss.backward()
            opt_zs_test.step()

    # Save stats and plots for report
    model.eval()
    log_final_stats(config, model)
    log_final_plots(config, model, train_loader, test_loader)
      
if __name__ == "__main__":
    # Store hyper-parameters
    config = hyperparameters()

    # Create path to save log files
    log_name = f"{config.model}_{config.data}_ld{config.lat_dim}"
    config.main_path = f'../log/{log_name}'

    if not os.path.exists(config.main_path):
        os.mkdir(config.main_path)

    # Run training and validation
    train(config)

