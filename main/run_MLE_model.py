
from HVAE import HVAE
from LVAE import LVAE

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import DataLoader, Dataset

import torch
import scipy.io
import numpy as np
import pandas as pd

import wandb
from tqdm import tqdm
from collections import defaultdict
import argparse
import os
from sklearn import decomposition
import seaborn as sns
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

os.environ["WANDB_API_KEY"] = "44b27736c17cd4318936992ac992d9d787f1a5e5"

train_data = defaultdict(list)
val1_data = defaultdict(list)
val2_data = defaultdict(list)

def get_hyperparameters():
    """ Description: Gets hyper-parameters from command line and stores all hyper-parameters in an object
        Return: config-object """

    # Get varying hyper-parameters from command line
    argparser = argparse.ArgumentParser(description='Get hyper-parameters')

    argparser.add_argument('-data',type=str,default="MNIST",help='dataset name')
    argparser.add_argument('-model',type=str,default="HVAE",help='model type')
    argparser.add_argument('-epochs',type=int,default=20,help='#training epochs')
    argparser.add_argument('-sl',type=int,default=5,help='#stochastic layers')
    argparser.add_argument('-s_train',type=int,default=1,help='#importance samples for training')
    argparser.add_argument('-s_test',type=int,default=20,help='#importance samples for extra test')
    argparser.add_argument('-wu',type=int,default=0,help='#warm-up epochs')
    argparser.add_argument('-bn',type=int,default=0,help='Apply batch normalization')

    args = argparser.parse_args()

    # Create hyper-parameter object
    class config():
        def __init__(self, data, model, epochs, sl, s_train, s_test, wu, bn):
            self.data = data
            self.model = model
            self.train_bs = 256
            self.test_bs = 256
            self.epochs = epochs
            self.sl = sl
            self.s_train = s_train
            self.s_test = s_test
            self.wu = wu
            self.lr = 1e-3
            self.bn = bn
            self.main_path = ""

    config = config(args.data, args.model, args.epochs, args.sl, args.s_train, args.s_test, args.wu, args.bn)

    return config

def load_data(config):
    """ Description: Load dataset and create data loaders
        Return: train_loader, test_loader
    """

    if config.data == "MNIST":
        # Transform function
        transform = Compose([
            ToTensor(),
            Lambda(lambda x: torch.flatten(x)),
            Lambda(lambda x: torch.bernoulli(x)),
        ])

        # Load train and test datasets
        data_train = MNIST("../data", train=True, transform=transform, download=True)
        data_test = MNIST("../data", train=False, transform=transform, download=True)

    if config.data == "OMNIGLOT":
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
    train_loader = DataLoader(data_train, batch_size=config.train_bs, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=config.test_bs, shuffle=True)

    return train_loader, test_loader

def load_model(config, train_loader):
    """ Description: Initialize model
        Return: Model
    """

    # Load batch of images
    images, labels = next(iter(train_loader))

    # Initialize model
    if config.model == "HVAE":
        model = HVAE(in_features=images.size(1), layers=config.sl,batch_norm=config.bn, device=device).to(device)

    if config.model == "LVAE":
        model = LVAE(in_features=images.size(1), layers=config.sl,batch_norm=config.bn, device=device).to(device)

    # Initialize weights with Glorot optimization (as in LVAE paper)
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)

    model.apply(init_weights)

    return model

def build_optimizer(config, model):
    """ Description: Initialize optimizer
        Return: Optimizer
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    return optimizer

def train_epoch(config, model, train_loader, optimizer, epoch):
    # Collect mean diagnostic data for each batch. It is a dict with a key for each diagnostic. The value for each key
    # is a list of batch means.
    training_batch_data = defaultdict(list)

    with tqdm(train_loader, unit="batch", desc=f" {epoch+1}/{config.epochs}") as batch:
        for x, y in batch:
            # Prepare for training
            x = x.to(device)
            optimizer.zero_grad()

            # Forward pass
            train_loss, diagnostics = model.VariationalInference(x, imp_k=config.s_train, warm_up=config.wu,
                                                                 epoch=epoch)

            # Optimization step
            train_loss.backward()
            optimizer.step()
            
            # Save batch mean for each statistic for each batch
            for k, v in diagnostics.items():
                training_batch_data[k] += [v.item()]

        # Save single epoch mean of batch means for each statistic
        for k, v in training_batch_data.items():
            train_data[k] += [np.mean(training_batch_data[k])]

def test_epoch(config, model, test_loader, epoch, val_type):
    # Collect mean diagnostic data for each batch. It is a dict with a key for each diagnostic. The value for each key
    # is a list of batch means.
    validation_batch_data = defaultdict(list)

    if (val_type == 2) and (epoch % 20 == 0):
        if epoch == 0:
            best_loss = 5000
        else:
            best_loss = min(val2_data['loss'][:-19])

    with torch.no_grad():
        for x, y in test_loader:
            # Prepare for validation
            x = x.to(device)

            # Forward pass
            if val_type == 1:
                imp_k = config.s_train
            if val_type == 2:
                imp_k = config.s_test

            val_loss, diagnostics = model.VariationalInference(x, imp_k=imp_k, warm_up=config.wu,
                                                               epoch=epoch)

            # Save batch mean for each statistic for each batch
            for k, v in diagnostics.items():
                validation_batch_data[k] += [v.mean().item()]

        # Save single epoch mean of batch means for each statistic and
        for k, v in validation_batch_data.items():
            if val_type == 1:
                val1_data[k] += [np.mean(validation_batch_data[k])]

            if val_type == 2:
                val2_data[k] += [np.mean(validation_batch_data[k])]

        # Check every 20th epoch if the model has become better. If so then save the model. Only do this if the script is
        # run as main script (e.g. don't do it during hyper-parameter search)
        if val_type == 2:
            if (epoch % 20 == 0) and (val2_data['loss'][-1] < best_loss):
                # Save model
                torch.save(model.state_dict(), f"{config.main_path}/model.pth")

                # Save number of best epoch
                lol = pd.Series()
                lol['best_epoch'] = epoch + 1
                lol.to_csv(f'{config.main_path}/best_epoch.csv')

def plot_wandb(model, test_loader):
    """ Part 1: Plot images generated from prior """
    x, y = next(iter(test_loader))
    x = x.to(device)

    # Get p(x|z) for z generated from prior p(z) and sample from it
    px = model.sample_from_prior(batch_size=x.size(0))['PxGz1']
    x_samples = px.mean

    # Plot images originating from 5 of the prior samples (wandb)
    images = [wandb.Image(x_samples[i, :].view(28, 28)) for i in range(5)]
    images = pd.DataFrame({"": images})

    image_table2 = wandb.Table(data=images)
    wandb.log({"Images generated from prior": image_table2})

    """ Part 2: Plot image and "reconstructed" image (testing data) """
    # Get data from test batch
    x, y = next(iter(test_loader))
    x = x.to(device)

    # Get observation model from x->z->x
    _, _, _, PxGz1 = model(x, imp_k=1)
    x_rec = PxGz1[0].mean

    # Collect original images and reconstructions
    images_A = [wandb.Image(x[i, :].view(28, 28)) for i in range(5)]
    images_B = [wandb.Image(x_rec[i, :].view(28, 28)) for i in range(5)]

    # Save as wandb-table
    images = pd.DataFrame({"Label": y[:5],
                           "Original image": images_A,
                           "Reconstructed image": images_B
                           })

    image_table1 = wandb.Table(data=images)
    wandb.log({"Original and reconstruction (test)": image_table1})

    """ Part 3: Latent representations (testing data) """
    ys = []
    zs = []
    count = 0

    # Use samples from multiple batches
    for x, y in test_loader:
        x = x.to(device)

        if isinstance(model, HVAE):
            z, _ = model.encode(x, imp_k=1)

        if isinstance(model, LVAE):
            z, _, _, _ = model.encode(x, imp_k=1)

        if count == 0:
            zs += z[0]
            count += 1
        else:
            zs = [torch.vstack((a,b)) for a,b in zip(zs,z[0])]

        y = y.tolist()
        ys += [str(element) for element in y]

    pca = decomposition.PCA(n_components=2)

    plots = []

    for i in range(model.layers):
        # Data for plotting
        x = zs[i].detach().cpu().numpy()

        # PCA on x
        pca.fit(x)
        x = pca.transform(x)

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
        image = wandb.Image(plt)
        plots.append(image)

    images = pd.DataFrame({"Layer": np.arange(1, model.layers + 1),
                           "PCA plot": plots,
                           })

    image_table2 = wandb.Table(data=images)
    wandb.log({"Latent representations": image_table2})

def log_final_stats(config, model, test_loader):
    """ Part 1: Calculate and save latent representation activation statistic """
    mus = []
    count = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            _, _, imp_qs, _ = model(x, imp_k=config.s_train)

            # Batch average
            mu = [0] * config.sl

            for i in range(config.s_train):
                for j in range(config.sl):
                    mu[j] += imp_qs[i][j].mean / config.s_train

            if count == 0:
                mus += mu
                count += 1

            else:
                for i in range(config.sl):
                    mus[i] = torch.vstack((mus[i], mu[i]))

    layer_stats = [torch.var(mus[i], dim=0) for i in range(len(mus))]
    above_treshold = [torch.sum(layer_stats[i] > 10 ** (-2)).detach().cpu().item() for i in range(len(mus))]

    # Save
    lol = pd.Series()
    lol['activation'] = above_treshold
    lol.to_csv(f'{config.main_path}/activation_stats.csv')

    """ Part 2: Logging loss """
    # Use dataframe as main log
    main_log = pd.DataFrame()

    # Log training epoch results
    main_log['train_loss'] = np.array(train_data['loss'])
    main_log['train_beta_loss'] = np.array(train_data['beta_loss'])
    main_log['train_rec'] = np.array(train_data['rec_comp'])

    # Log test 1 (importance sampling as in training) epoch results
    main_log['val1_loss'] = np.array(val1_data['loss'])
    main_log['val1_beta_loss'] = np.array(val1_data['beta_loss'])
    main_log['val1_rec'] = np.array(val1_data['rec_comp'])

    # Log test 2 (high number of importance samples) epoch results
    main_log['val2_loss'] = np.array(val2_data['loss'])
    main_log['val2_beta_loss'] = np.array(val2_data['beta_loss'])
    main_log['val2_rec'] = np.array(val2_data['rec_comp'])

    main_log.to_csv(f'{config.main_path}/main_log.csv')

def log_final_plots(config, model, train_loader, test_loader):
    """ Part 1: Plot decoded samples from prior """
    x, y = next(iter(test_loader))
    x = x.to(device)

    # Get p(x|z) for z generated from prior p(z) and sample from it
    px = model.sample_from_prior(batch_size=x.size(0))['PxGz1']
    x_samples = px.mean

    for i in range(30):
        x_sample = x_samples[i, :].view(28, 28)

        # Create plot
        fig, ax = plt.subplots()
        ax.imshow(x_sample.detach().cpu(), cmap='gray')

        # Save plot
        save_path = f'{config.main_path}/priors'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        fig.savefig(f'{save_path}/prior{i}')
        plt.close()

    """ Part 2: Reconstruction plot (training) """
    x, y = next(iter(train_loader))
    x = x.to(device)

    # Get observation model from x->z->x
    _, _, _, PxGz1 = model(x, imp_k=1)
    x_rec = PxGz1[0].mean

    for i in range(30):
        reconstruction = x_rec[i, :].view(28, 28)
        original = x[i, :].view(28, 28)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(original.cpu(), cmap='gray')
        ax2.imshow(reconstruction.detach().cpu(), cmap='gray')

        # Save
        save_path = f'{config.main_path}/train_recs'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        fig.savefig(f'{save_path}/rec{i}')
        plt.close()

    """ Part 3: Plot latent representations (training) """
    ys = []
    zs = []
    count = 0

    # Use samples from multiple batches
    for x, y in train_loader:
        x = x.to(device)

        if isinstance(model, HVAE):
            z, _ = model.encode(x, imp_k=1)

        if isinstance(model, LVAE):
            z, _, _, _ = model.encode(x, imp_k=1)

        if count == 0:
            zs += z[0]
            count += 1
        else:
            zs = [torch.vstack((a, b)) for a, b in zip(zs, z[0])]

        y = y.tolist()
        ys += [str(element) for element in y]

    pca = decomposition.PCA(n_components=2)

    for i in range(model.layers):
        # Data for plotting
        x = zs[i].detach().cpu().numpy()

        # PCA on x
        pca.fit(x)
        x = pca.transform(x)

        # Structure data and plot it
        df = pd.DataFrame(dict(x1=x[:, 0], x2=x[:, 1], y=ys))

        fig, ax = plt.subplots()

        if config.data == "OMNIGLOT":
            sns.scatterplot(x='x1', y='x2', data=df, hue='y', size=3, alpha=0.2, legend=False)
        else:
            colors = {'0': 'black', '1': 'red', '2': 'green', '3': 'yellow', '4': 'brown', '5': 'cyan', '6': 'pink',
                      '7': 'orange', '8': 'chocolate', '9': 'darkkhaki'}
            sns.scatterplot(x='x1', y='x2', data=df, hue='y', palette=colors, size=3, alpha=0.2)

        # Save plot
        save_path = f'{config.main_path}/train_lat_reps'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        fig.savefig(f'{save_path}/layer{i}')
        plt.close()

    """ Part 4: Reconstruction plot (testing) """
    x, y = next(iter(test_loader))
    x = x.to(device)

    # Get observation model from x->z->x
    _, _, _, PxGz1 = model(x, imp_k=1)
    x_rec = PxGz1[0].mean

    for i in range(30):
        reconstruction = x_rec[i, :].view(28, 28)
        original = x[i, :].view(28, 28)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(original.cpu(), cmap='gray')
        ax2.imshow(reconstruction.detach().cpu(), cmap='gray')

        # Save
        save_path = f'{config.main_path}/test_recs'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        fig.savefig(f'{save_path}/rec{i}')
        plt.close()

    """ Part 5: Plot latent representations (testing) """
    ys = []
    zs = []
    count = 0

    # Use samples from multiple batches
    for x, y in test_loader:
        x = x.to(device)

        if isinstance(model, HVAE):
            z, _ = model.encode(x, imp_k=1)

        if isinstance(model, LVAE):
            z, _, _, _ = model.encode(x, imp_k=1)

        if count == 0:
            zs += z[0]
            count += 1
        else:
            zs = [torch.vstack((a, b)) for a, b in zip(zs, z[0])]

        y = y.tolist()
        ys += [str(element) for element in y]

    pca = decomposition.PCA(n_components=2)

    for i in range(model.layers):
        # Data for plotting
        x = zs[i].detach().cpu().numpy()

        # PCA on x
        pca.fit(x)
        x = pca.transform(x)

        # Structure data and plot it
        df = pd.DataFrame(dict(x1=x[:, 0], x2=x[:, 1], y=ys))

        fig, ax = plt.subplots()

        if config.data == "OMNIGLOT":
            sns.scatterplot(x='x1', y='x2', data=df, hue='y', size=3, alpha=0.2, legend=False)
        else:
            colors = {'0': 'black', '1': 'red', '2': 'green', '3': 'yellow', '4': 'brown', '5': 'cyan', '6': 'pink',
                      '7': 'orange', '8': 'chocolate', '9': 'darkkhaki'}
            sns.scatterplot(x='x1', y='x2', data=df, hue='y', palette=colors, size=3, alpha=0.2)

        # Save plot
        save_path = f'{config.main_path}/test_lat_reps'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        fig.savefig(f'{save_path}/layer{i}')
        plt.close()

def train(config):
    # Set up weight and biases logging (fast-prototyping)
    wandb.init(config=config, project="bachelor_project", entity="louisdt")
    wandb.run.name = f"{config.data} - {config.model}"

    # Load data and pre-process data
    train_loader, test_loader = load_data(config)

    # Load model
    model = load_model(config, train_loader)

    # Load optimizer
    optimizer = build_optimizer(config, model)

    # Train and evaluate model
    for epoch in range(config.epochs):
        # Train model 1 epoch (with chosen number importance sample)
        model.train()
        train_epoch(config, model, train_loader, optimizer, epoch)

        # Test model on 1 epoch (with same number of importance sample)
        model.eval()
        test_epoch(config, model, test_loader, epoch, 1)

        # Test model on high number of epochs
        model.eval()
        test_epoch(config, model, test_loader, epoch, 2)

        # Prototyping plots
        if epoch%20 == 0:
            plot_wandb(model, test_loader)

        # wandb logging
        data = {
            'train_loss': train_data['loss'][-1],
            'train_beta_loss': train_data['beta_loss'][-1],
            'train_rec': train_data['rec_comp'][-1],
            'val1_loss': val1_data['loss'][-1],
            'val1_beta_loss': val1_data['beta_loss'][-1],
            'val1_rec': val1_data['rec_comp'][-1],
            'val2_loss': val2_data['loss'][-1],
            'val2_beta_loss': val2_data['beta_loss'][-1],
            'val2_rec': val2_data['rec_comp'][-1],
        }

        wandb.log(data)

    # Load best model
    model.eval()
    model.load_state_dict(torch.load(f"{config.main_path}/model.pth"))

    # Save stats and plots for report
    log_final_stats(config, model, test_loader)
    log_final_plots(config, model, train_loader, test_loader)

if __name__ == "__main__":
    # Get hyper-parameters and store in object
    config = get_hyperparameters()

    # Creat path for results (main log) and plots for report
    log_name = f"{config.model}_{config.data}_sl{config.sl}_strain{config.s_train}_wu{config.wu}_bn{config.bn}"
    config.main_path = f'../log/{log_name}'

    if not os.path.exists(config.main_path):
        os.mkdir(config.main_path)

    # Run training and validation
    train(config)

