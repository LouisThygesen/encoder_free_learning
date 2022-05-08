# FINISHED: DO NOT CHANGE ANYTHING

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions import Bernoulli
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.uniform import Uniform
from torch.distributions.independent import Independent

# Set up GPU support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class DGD_GMM(nn.Module):
    def __init__(self, config, train_size, test_size):
        super(DGD_GMM, self).__init__()

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

        # Define latent representations as parameter
        self.zs_train = Parameter(torch.normal(0, 1, size=(train_size, self.latent_dim)))
        self.zs_test = Parameter(torch.normal(0, 1, size=(test_size, self.latent_dim)))

        # p(\phi) distributions
        self.dist1 = Dirichlet(torch.ones(10).to(device))
        self.dist3 = torch.distributions.Independent(Normal(torch.ones(10,self.latent_dim).to(device)*2,
                                                            torch.ones(10,self.latent_dim).to(device)*2),1)

        # \phi parameters (initally sampled from priors)
        initializer = Uniform(low=torch.ones(10,self.latent_dim).to(device)*(-5),
                              high=torch.ones(10,self.latent_dim).to(device)*5)

        self.mixture_coeffs = Parameter(self.dist1.sample())
        self.Gaussian_mean = Parameter(initializer.sample())
        self.Gaussian_other = Parameter(self.dist3.sample())

    def prior(self, batch_size):
        # p(z|\phi) (Gaussian mixture model)
        mix = Categorical(nn.functional.softmax(self.mixture_coeffs))
        comp = Independent(Normal(self.Gaussian_mean,(1/self.Gaussian_other.exp())**0.5), 1)
        gmm = MixtureSameFamily(mix, comp)

        # Expand to batch size
        gmm = gmm.expand((batch_size,))

        return gmm

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
        log_Pzk = pz.log_prob(zs) # log(p(z))

        # Prior probability of representation and probability of x given representation
        log_PxGz = log_PxkGzk.sum()
        log_Pz = log_Pzk.sum()

        # Caculate loss from \phi
        log_prob1 = self.dist1.log_prob(nn.functional.softmax(self.mixture_coeffs,dim=0))
        log_prob3 = self.dist3.log_prob(self.Gaussian_other).sum()

        phi_loss = log_prob1 + log_prob3

        # TODO: Theta prior does not work here (does not correct parameter groups)!
        # If weights are initialized using N(0,I) and a prior over the network weights is included in the model
        if self.theta_prior == 1:
            # MLP weights loss calculation
            weights1 = [pam for pam in self.parameters()][1::2]
            weights2 = [i.view(-1,1).squeeze(1) for i in weights1]
            weights3 = weights2[0]
        
            for i in range(1,3):
                weights3 = torch.concat((weights3,weights2[i]))
        
            dist = Normal(loc=torch.zeros(weights3.size(0)).to('cuda'), scale=torch.ones(weights3.size(0)).to('cuda'))
            theta_loss = dist.log_prob(weights3).sum()
        
            # Final probability P(X,Z)
            log_Pxz = log_PxGz + log_Pz + phi_loss + theta_loss
            
        else:
            # Final probability P(X,Z)
            log_Pxz = log_PxGz + log_Pz + phi_loss
        
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
        z = Pz.sample()

        # Send latent representation through decoder
        probs = self.mlp(z)

        # Define P(x|z)
        PxGz = Bernoulli(probs=probs)

        return {'PxGz': PxGz, 'Pz': Pz, 'z': z}

    def reset_zs_test(self):
        self.zs_test = Parameter(torch.normal(0, 1, size=(self.test_size, self.latent_dim)).to(device))




