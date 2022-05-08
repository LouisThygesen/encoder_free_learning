
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from torch.distributions import Normal

class encoder_block(nn.Module):
    def __init__(self,in_features,out_features,img_size, batch_norm):
        super().__init__()

        # Needed in forward pass
        self.in_features = in_features
        self.out_features = out_features
        self.batch_norm = batch_norm
        self.img_size = img_size

        # Set up main MLP
        if batch_norm:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features*4),
                nn.LeakyReLU(),
                nn.Linear(in_features=out_features*4, out_features=out_features*4),
                nn.BatchNorm1d(out_features * 4),
                nn.LeakyReLU(),
                nn.Linear(in_features=out_features*4, out_features=out_features*2)
            )

        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features*4),
                nn.LeakyReLU(),
                nn.Linear(in_features=out_features*4, out_features=out_features*4),
                nn.LeakyReLU(),
                nn.Linear(in_features=out_features*4, out_features=out_features*2)
            )

    def forward(self,x,imp_k=None):
        # Through encoder block network
        d = self.mlp(x)
        mu = d[:, 0:self.out_features]
        log_var = d[:, self.out_features:]

        # Decoder block distribution
        QzGx = Normal(mu, (0.5 * log_var).exp())

        # Sample imp_k importance samples
        if self.in_features == self.img_size:
            z = QzGx.rsample(torch.Size([imp_k]))
        else:
            z = QzGx.rsample()

        return QzGx, z

class decoder_block(nn.Module):
    def __init__(self,in_features,out_features,img_size,batch_norm):
        super().__init__()

        self.batch_norm = batch_norm
        self.out_features = out_features
        self.img_size = img_size

        if out_features != self.img_size:
            if batch_norm:
                # MLP
                self.mlp = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=in_features * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=in_features * 4, out_features=in_features * 4),
                    nn.BatchNorm1d(in_features * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=in_features * 4, out_features=out_features * 2),
                )
            else:
                # MLP
                self.mlp = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=in_features * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=in_features * 4, out_features=in_features * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=in_features * 4, out_features=out_features * 2),
                )

        else:
            # MLP
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

        if self.out_features != self.img_size:
            # Obtain parameters
            mu = d[:, 0:self.out_features]
            log_var = d[:, self.out_features:]

            # Define distribution
            PxGz = Normal(mu, (0.5 * log_var).exp())

            # Sample latent represntation
            z = PxGz.rsample()

        else:
            # Define distribution
            PxGz = Bernoulli(logits=d)

            # Sample latent represntation
            z = PxGz.sample()

        return PxGz, z

class HVAE(nn.Module):
    def __init__(self, in_features, layers, batch_norm, device):
        super(HVAE, self).__init__()

        # Needed in forward pass
        self.layers = layers
        self.batch_norm = batch_norm
        self.device = device
        self.img_size = in_features

        # List of number input features and latent representation features (5 layers: [img_size,64,32,16,8,4])
        self.features= [in_features] + [int(64/(2**layer)) for layer in range(layers)]

        # List of encoder block modules for inference model
        self.encoder = nn.ModuleList([encoder_block(self.features[i],self.features[i+1],self.img_size,batch_norm) for i in range(layers)])

        # List of decoder block modules for generative model
        self.decoder = nn.ModuleList([decoder_block(self.features[i],self.features[i-1],self.img_size,batch_norm) for i in range(layers,0,-1)])

        # Define prior parameters in buffer (not trainable)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2 * self.features[-1]])))

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

    def encode(self, x, imp_k):
        """ Part 1: Importance sampling data collection """
        # Importance weighted data: List of lists where each inner list is going to describe the data of an importance
        # sampling path from z_1 to z_L.
        imp_qs = []
        imp_zs = []

        """ Part 2: Data through network """
        # Send input through first encoder block (with importance sampling) to get 1 shared distribution Pz1Gx and
        # k_imp importance samples of z1 (stored in a list)
        Pz1Gx, z1s = self.encoder[0](x,imp_k)

        # Use each z1 as the seed for a importance sampling path - send each importance sample through the rest of the
        # encoder blocks (without further sampling)
        for sample in range(imp_k):
            # Importance sample seed
            z = z1s[sample]

            # Path representations and distributions
            zs = [z]
            qs = [Pz1Gx]

            for block in range(1,len(self.encoder)):
                # Get q(z|x) for current z and x
                dist, z = self.encoder[block](z)

                # Save distribution and latent representation
                zs += [z]
                qs += [dist]

            # Add to importance sampling data collection
            imp_qs += [qs]
            imp_zs += [zs]

        return imp_zs, imp_qs

    def decode(self, imp_zs, imp_k):
        """ Part 1: Importance sampling data collection """
        # Importance weighted data: List of lists where each inner list describes the data for an importance sampling
        # path from z_1 to z_L.
        imp_first_dists = []
        imp_last_dist = []

        """ Part 2: Data through network """
        # Send each importance sample through the decoder blocks (without further importance sampling)
        for sample in range(imp_k):
            # Get all decoder latent representations for importance sample
            zs = imp_zs[sample]

            # Save decoder distributions (local save)
            first_dists = []

            # Send latent representations from the encoder through all but the last decoder block
            if len(self.decoder) > 1:
                for (block, z) in zip(self.decoder[:-1], zs[::-1]):
                    # Get p(x|z) for current x and z
                    dist, _ = block(z)

                    # Save distribution - not latent representation (local save)
                    first_dists += [dist]

            # Send latent representation from the encoder through the last decoder block (local save)
            last_dist, _ = self.decoder[-1](zs[0])

            # Add to importance sampling data collection (global save)
            imp_first_dists += [first_dists]
            imp_last_dist += [last_dist]

        return imp_first_dists, imp_last_dist

    def forward(self, x, imp_k):
        # Define prior distribution for z_L (innermost latent representation)
        PzL = [self.prior(batch_size=x.size(0))]

        # Run enccoder blocks (with importance sampling)
        imp_zs, imp_qs = self.encode(x,imp_k)

        # Run decoder blocks
        imp_first_ps, imp_last_p = self.decode(imp_zs,imp_k)

        # For each important sample save the prior distribution along with the other decoder distributions (except
        # the last decoder distribution q(x|z_1) which is saved separately). The goal with this is to be able to
        # calculate the loss easily.
        imp_other_ps = []

        for dists in imp_first_ps:
            temp = PzL + dists
            imp_other_ps += [temp]

        return imp_zs, imp_other_ps, imp_qs, imp_last_p

    def VariationalInference(self, x, imp_k, warm_up, epoch):
        # Forward pass through model to obtain encoder and decoder distributions as well as latent variables (from
        # decoder) for all importance samples. The data is list of lists (outer index = importance sample)
        imp_zs, imp_other_ps, imp_qs, imp_last_p = self.forward(x, imp_k)

        # Current beta value (linear interpolation up to last 'warm-up' epoch)
        if warm_up:
            beta = min(1.0 / warm_up * epoch, 1.0)
        else:
            beta = 1.0

        # Calculate for each importance sample path
        batch_size = imp_zs[0][0].size(0)
        imp_elbo = torch.zeros(imp_k, batch_size).to(self.device)
        imp_beta_elbo = torch.zeros(imp_k, batch_size).to(self.device)

        for k in range(imp_k):
            """ Caluclate probabilities for events from all encoder and decoder distributions (using log rules) """
            # In case of 3 layers: [Pz3(z3), Pz2Gz3(z2), Pz1Gz2(z1)]
            log_probs1 = [imp_other_ps[k][i].log_prob(imp_zs[k][::-1][i]).sum(dim=1) for i in range(self.layers)]

            # In case of 3 layers: [Qz1Gx(z1), Qz2Gz1(z2), Qz3Gz2(z3)]
            log_probs2 = [imp_qs[k][i].log_prob(imp_zs[k][i]).sum(dim=1) for i in range(self.layers)]

            # In case of any number of layers: [PxGz1(x)]
            log_probs3 = imp_last_p[k].log_prob(x).sum(dim=1)

            """ Caluclate final probabilities by combining the probabilities from above (using log rules)  """
            # In case of 3 layers: log_Pz = log_Pz3 + log_Pz2Gz3 + log_Pz1Gz2
            log_Pz = sum(log_probs1)

            # In case of 3 layers: log_QzGx = log_Qz1Gx + log_Qz2Gz1 + log_Qz3Gz2
            log_QzGx = sum(log_probs2)

            # In case of any number of layers: log_PxGz = log_PxGz1
            log_PxGz = log_probs3

            # Compute ELBO and beta-ELBO (for each batch)
            kl = log_QzGx - log_Pz
            elbo = log_PxGz - kl
            beta_elbo = log_PxGz - beta * kl

            # Save results
            imp_elbo[k,:] = elbo
            imp_beta_elbo[k,:] = beta_elbo

            if k==0:
                rec_loss = log_PxGz

        # Calulate importance weighted statistics for each data point in batch
        log_ks = (torch.ones(beta_elbo.size(0)) * torch.Tensor([imp_k]).log()).to(self.device)

        iw_elbo = imp_elbo.logsumexp(dim=0) - log_ks
        iw_beta_elbo = imp_beta_elbo.logsumexp(dim=0) - log_ks

        # The mini-batch loss
        loss = -iw_beta_elbo.mean()

        # Log results
        with torch.no_grad():
            diagnostics = {'loss': -iw_elbo.mean(),
                           'beta_loss': -iw_beta_elbo.mean(),
                           'rec_comp': -rec_loss.mean(),
            }

        return loss, diagnostics

    def sample_from_prior(self, batch_size):
        # Define and sample the prior distribution P(z_L)
        PzL = self.prior(batch_size=batch_size)
        z = PzL.rsample()

        # Define the observation model p(x|z) = B(x | g(z)). Notice I need not wait with sending the signal through the
        # last decoder block as I don't need to save the distributions in a special way for a loss function (as in the
        # encoder function).
        for block in self.decoder:
            # Get p(x|z) for current x and z
            PxGz, z = block(z)

        return {'PxGz1': PxGz, 'PzL': PzL, 'z': z}
