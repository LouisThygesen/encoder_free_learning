# FINISHED: DO NOT CHANGE ANYTHING

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions import Bernoulli
from torch.distributions import Normal

# Set up GPU support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class decoder_block(nn.Module):
    def __init__(self, in_features, out_features, batch_norm):
        super().__init__()

        self.out_features = out_features

        if out_features != 28 ** 2:
            if batch_norm:
                self.mlp = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=in_features * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=in_features * 4, out_features=in_features * 4),
                    nn.BatchNorm1d(in_features * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=in_features * 4, out_features=out_features),
                )
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=in_features * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=in_features * 4, out_features=in_features * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=in_features * 4, out_features=out_features),
                )

        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_features*4),
                nn.LeakyReLU(),
                nn.Linear(in_features=in_features*4, out_features=in_features*4),
                nn.LeakyReLU(),
                nn.Linear(in_features=in_features * 4, out_features=out_features),
            )

    def forward(self,x):
        # Send input through MLP
        d = self.mlp(x)

        if self.out_features != 28 ** 2:
            # Obtain parameters
            mu = d
            
            # Define distribution
            PxGz = Normal(mu, 1/4)
            #PxGz = Normal(mu, (1/log_var.exp())**0.5)

            # Sample latent represntation
            z = PxGz.rsample()

        else:
            # Define distribution
            PxGz = Bernoulli(logits=d)

            # Sample latent represntation
            z = PxGz.sample()

        return PxGz, z

class HDGD(nn.Module):
    def __init__(self, config, train_size, test_size):
        super(HDGD, self).__init__()

        # Information needed in forward pass and loss
        self.layers = config.sl
        self.test_size = test_size
        self.theta_prior = config.theta_prior 

        # List of number of input/outout features (e.g. with 5 layers: [4,8,16,32,64,784])
        reverse_list = [784] + [int(config.lat_dim / (2 ** layer)) for layer in range(self.layers)]  
        self.features = reverse_list[::-1]

        # Decoder is defined as a list of decoder blocks
        self.decoder = nn.ModuleList([])

        for i in range(self.layers):
            self.decoder.append(decoder_block(self.features[i], self.features[i + 1], config.bn))

        # Latent space prior p(z_L)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2 * self.features[0]])))

        # Latent representations for training and testing
        self.zs_train = []
        self.zs_test = []

        for i in range(self.layers):
            self.zs_train += [torch.normal(0, 1, size=(train_size, self.features[i])).to(device)]
            self.zs_test += [torch.normal(0, 1, size=(test_size, self.features[i])).to(device)]

        self.zs_train = [Parameter(i) for i in self.zs_train]
        self.zs_test = [Parameter(i) for i in self.zs_test]

    def prior(self, batch_size):
        # The prior parameters (vector of size latent_dim * 2) is expanded to a matrix of size
        # (batch_size, latent_dim * 2) where latent_dim is the dimension of the of latent
        # representation L
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])

        # The matrix is split into 2 matrices of size (batch_size,latent_dim). In the case latent_dim=2, the 2 matrices
        # contains respectively parameters (mu1,mu2) and (log_sigma1,log_sigma2) for each image in the batch
        mu, log_sigma = prior_params.chunk(2, dim=-1)

        # Use epx(0) = 1
        sigma = log_sigma.exp()

        # Return prior distribution
        return Normal(mu, sigma)

    def forward(self, x, idx, mode):
        # Define prior distribution for z_L (innermost latent representation)
        Pz = self.prior(batch_size=x.size(0))

        # Get latent representations
        if mode == "train":
            zs = [element[idx] for element in self.zs_train]
        if mode == "test":
            zs = [element[idx] for element in self.zs_test]

        # Distribution group
        group1 = [Pz]

        for i in range(len(self.decoder)-1):
            # Get latent representations for layer
            zs_layer = zs[i]

            # Send latent representation through decoder block
            dist, _ = self.decoder[i](zs_layer)
            group1 += [dist]

        # Send z_1 through last decoder block
        zs_layer = zs[-1]
        dist, _ = self.decoder[-1](zs_layer)

        return group1, dist

    def loss(self, x, idx, mode):
        # Forward pass through the model
        group1, dist = self.forward(x, idx, mode)

        # Get latent representations
        if mode == "train":
            zs = [element[idx] for element in self.zs_train]
        if mode == "test":
            zs = [element[idx] for element in self.zs_test]

        # Evaluate log probabilities for batch images and latent representations
        log_PxGz1 = dist.log_prob(x).sum(dim=1)
        log_other = [group1[i].log_prob(zs[i]).sum(dim=1) for i in range(self.layers)]

        # Calculate batch reconstruction and prior loss (normalized with batch size)
        term1 = log_PxGz1
        term2 = sum(log_other)

        # TODO: Theta prior does not work here (does not correct parameter groups)!
        # If weights are initialized using N(0,I) and a prior over the network weights is included in the model
        if self.theta_prior == 1:
            # MLP weights loss calculation
            weights1 = [pam for pam in self.parameters()][1::2]
            weights2 = [i.view(-1,1).squeeze(1) for i in weights1]
            weights3 = weights2[0]
        
            for i in range(1,3):
                weights3 = torch.cat((weights3,weights2[i]))
        
            dist = Normal(loc=torch.zeros(weights3.size(0)).to('cuda'), scale=torch.ones(weights3.size(0)).to('cuda'))
            theta_loss = dist.log_prob(weights3).sum()
        
            # Final probability P(X,Z)
            log_Pxz = term1 + term2 + theta_loss
        else:
            log_Pxz = term1 + term2
        
        # Final loss
        loss = - log_Pxz.mean()

        # Save results
        save1 = -term1.mean()
        save2 = -log_other[0].mean()
        save3 = loss - save1 - save2
        
        with torch.no_grad():
            diagnostics = {'loss': loss,
                           'log_PxGz': save1,
                           'log_PzL': save2,
                           'log_rest': save3}

        return loss, diagnostics

    def sample_from_prior(self, batch_size):
        # Define the prior p(z)
        PzL = self.prior(batch_size=batch_size)

        # Sample the prior
        x = PzL.rsample()

        # Send latent representation through decoder
        for block in self.decoder:
            # Get p(x|z) for current x and z
            PxGz, x = block(x)

        return {'PxGz': PxGz, 'PzL': PzL}

    def reset_zs_test(self):
        self.zs_test = []

        for i in range(self.layers):
            self.zs_test += [torch.normal(0, 1, size=(self.test_size, self.features[i])).to(device)]

        self.zs_test = [Parameter(i) for i in self.zs_test]
