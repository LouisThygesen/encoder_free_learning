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

class DGD(nn.Module):
    def __init__(self, config, train_size, test_size):
        super(DGD, self).__init__()

        # Information needed in forward pass and loss
        self.latent_dim = config.lat_dim
        self.out_dim = 28**2
        self.test_size = test_size
        self.theta_prior = config.theta_prior 

        # Decoder network
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

        # Define prior parameters as buffer (not trainable)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2 * self.latent_dim])))

        # Define latent representations as parameter
        self.zs_train = Parameter(torch.normal(0, 1, size=(train_size, self.latent_dim)))
        self.zs_test = Parameter(torch.normal(0, 1, size=(test_size, self.latent_dim)))

    def prior(self, batch_size):
        """ Return the distribution p(z) """

        # The prior parameters (vector of size latent_shape * 2) is expanded to a matrix of size
        # (batch_size, latent_shape * 2)
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])

        # The matrix is split into as many matrices as the latent_shape. These matrices are of size
        # (batch_size,2). In the case latent_shape=2, the 2 matrices contains respectively parameters
        # (mu1,mu2) and (log_sigma1,log_sigma2) for each image in the batch
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        sigma = log_sigma.exp()

        return Normal(mu, sigma)

    def forward(self, x, idx, mode):
        # Define the prior distribution p(z)
        Pz = self.prior(batch_size=x.size(0))

        # Get latent representations
        if mode == "train":
            zs = self.zs_train[idx]
        if mode == "test":
            zs = self.zs_test[idx]

        # Send latent representation through decoder
        zs = self.mlp(zs)

        # Define P(x|z)
        PxGz = Bernoulli(probs = zs)

        return PxGz, Pz

    def loss(self, x, idx, mode):
        # Forward pass through the model
        px, pz = self.forward(x, idx, mode)

        if mode == "train":
            zs = self.zs_train[idx]
        if mode == "test":
            zs = self.zs_test[idx]

        # Evaluate log probabilities for batch images and latent representations
        # (not whole dataset) using rule log rule log(a)+log(b)=log(a*b)
        log_PxkGzk = px.log_prob(x).sum(dim=1)  # log(p(x|z))
        log_Pzk = pz.log_prob(zs).sum(dim=1)  # log(p(z))

        # Prior probability of representation and probability of x given representation
        log_PxGz = log_PxkGzk.sum()
        log_Pz = log_Pzk.sum()
        
        # If weights are initialized using N(0,I) and a prior over the network weights is included in the model
        if self.theta_prior == 1:
            # MLP weights loss calculation
            weights1 = [pam for pam in self.parameters()][1::2]
            weights2 = [i.view(-1,1).squeeze(1) for i in weights1]
            weights3 = weights2[0]

            for i in range(1,3):
                weights3 = torch.cat((weights3,weights2[i]))

            dist = Normal(loc=torch.zeros(weights3.size(0)).to(device), scale=torch.ones(weights3.size(0)).to(device))
            theta_loss = dist.log_prob(weights3).sum()
            theta_loss_normalized = theta_loss / dist.log_prob(weights3).size(0)

            # Final probability P(X,Z)
            log_Pxz = log_PxGz + log_Pz + theta_loss_normalized 
        else:
            # Final probability P(X,Z)
            log_Pxz = log_PxGz + log_Pz

        # Final loss
        loss = - log_Pxz 

        # Save output
        with torch.no_grad():
            diagnostics = {'loss': loss/x.size(0),
                           'log_PxGz': -log_PxGz/x.size(0),
                           'log_Pz': -log_Pz/x.size(0),
            }

        return loss, diagnostics

    def sample_from_prior(self, batch_size):
        # Define the prior p(z)
        Pz = self.prior(batch_size=batch_size)

        # Sample the prior
        z = Pz.rsample()

        # Send latent representation through decoder
        probs = self.mlp(z)

        # Define P(x|z)
        PxGz = Bernoulli(probs=probs)

        return {'PxGz': PxGz, 'Pz': Pz, 'z': z}

    def reset_zs_test(self):
        self.zs_test = Parameter(torch.normal(0, 1, size=(self.test_size, self.latent_dim)).to(device))