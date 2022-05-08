# FINISHED: DO NOT CHANGE ANYTHING

import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from torch.distributions import Normal

# Set up GPU support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class encoder_mlp(nn.Module):
    def __init__(self,in_features,out_features,batch_norm):
        super().__init__()

        # Set up main MLP
        if batch_norm:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.LeakyReLU(),
                nn.Linear(in_features=out_features, out_features=out_features),
                nn.BatchNorm1d(num_features=out_features),
                nn.LeakyReLU(),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.LeakyReLU(),
                nn.Linear(in_features=out_features, out_features=out_features),
                nn.LeakyReLU(),
            )

        # Parameter prediction layers (Bemærk varians - ikke std som tidligere)
        self.fcl1 = nn.Linear(in_features=out_features, out_features=int(out_features/4))
        self.fcl2 = nn.Sequential(
            nn.Linear(in_features=out_features, out_features=int(out_features/4)),
        )

    def forward(self,x):
        # Send input through MLP
        d = self.mlp(x)

        # Obtain parameters
        mu = self.fcl1(d)
        var = (0.5 * self.fcl2(d)).exp() + 1e-6

        return d, mu, var

class decoder_block(nn.Module):
    def __init__(self,in_features,out_features,img_size,batch_norm):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.img_size = img_size

        # Set up MLP and parameter prediction layers so that if we want batch norm it is applied to all expcept the last
        # decoder block
        if out_features != self.img_size:
            # Set up main MLP
            if batch_norm:
                self.mlp = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=in_features * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=in_features * 4, out_features=in_features * 4),
                    nn.BatchNorm1d(num_features=in_features * 4),
                    nn.LeakyReLU(),
                )
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=in_features * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=in_features * 4, out_features=in_features * 4),
                    nn.LeakyReLU(),
                )

            # Parameter prediction layers
            self.fcl1 = nn.Linear(in_features=in_features*4, out_features=out_features)
            self.fcl2 = nn.Sequential(
                nn.Linear(in_features=in_features*4, out_features=out_features),
            )

        else:
            # Set up main MLP
            self.mlp = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_features*4),
                nn.LeakyReLU(),
                nn.Linear(in_features=in_features*4, out_features=in_features*4),
                nn.LeakyReLU(),
            )

            # Parameter prediction layer
            self.fcl3 = nn.Linear(in_features=in_features*4, out_features=out_features)

    def forward(self,x):
        # Send input through MLP
        d = self.mlp(x)

        if self.out_features != self.img_size:
            # Obtain parameters
            mu = self.fcl1(d)
            var = (0.5 * self.fcl2(d)).exp() + 1e-6

            return mu, var

        else:
            # Obtain parameters
            logits = self.fcl3(d)

            return logits

class LVAE(nn.Module):
    def __init__(self, in_features, layers, batch_norm, device):
        super(LVAE, self).__init__()

        self.layers = layers
        self.device = device
        self.img_size = in_features

        # List of number input features and output features of MLPs (5 layers: [img_size, 256, 128, 64, 32, 16])
        self.features1 = [in_features] + [int(256/(2**layer)) for layer in range(layers)]
        self.features2 = [in_features] + [int(64/(2 ** layer)) for layer in range(layers)]

        # List of encoder block modules for inference model
        self.encoder = nn.ModuleList([encoder_mlp(self.features1[i], self.features1[i + 1],batch_norm) for i in range(layers)])

        # List of decoder block modules for generative model
        self.decoder = nn.ModuleList([decoder_block(self.features2[i],self.features2[i-1],self.img_size,batch_norm) for i in range(layers,0,-1)])

        # Define prior parameters in buffer (not trainable)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2 * self.features2[-1]])))

    def prior(self, batch_size):
        # The prior parameters (vector of size latent_shape * 2) is expanded to a matrix of size
        # (batch_size, latent_shape * 2)
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])

        # The matrix is split into as many matrices as the latent_shape. These matrices are of size
        # (batch_size,2). In the case latent_shape=2, the 2 matrices contains respectively parameters
        # (mu1,mu2) and (log_sigma1,log_sigma2) for each image in the batch
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        sigma = log_sigma.exp()

        # Return distribution `p(z)` (standard Gaussian)
        return torch.distributions.Normal(mu, sigma)

    def encode(self, x, imp_k):
        """ Part 1: Deterministic upward-pass (no importance sampling) """
        # Send input through all MLP blocks to obtain hat-parameters
        mu_q_hats = []
        var_q_hats = []

        for block in self.encoder:
            x, mu_q_hat, var_q_hat = block(x)

            # Save hat-parameters
            mu_q_hats += [mu_q_hat]
            var_q_hats += [var_q_hat]

        """ Part 2: Importance sampling data collection """
        # Importance weighted data: List of lists where each inner list is going to describe the data of an importance
        # sampling path from z_1 to z_L.
        imp_qs = []
        imp_zs = []

        # Importance weighted data: List where each element is the result of an importance sample
        mu_p_Lm1 = []
        var_p_Lm1 = []

        """ Part 3: Stochastic downward-pass (with importance sampling) """
        # Innermost latent distribution z_L is determined by corresponding hat-parameters and not precision-weighted
        # estimate (to get ball rolling).
        QzLGx = Normal(mu_q_hat, var_q_hat ** 0.5 + 1e-6)

        # Sample z_L multiple times (importance sampling)
        zLs = QzLGx.rsample(torch.Size([imp_k]))

        # Create reverse list of parameters
        mu_q_hats = mu_q_hats[::-1]
        var_q_hats = var_q_hats[::-1]

        # Loop over each z_L to send it through stochastic downward-pass (each creates an importane sampling path)
        for sample in range(imp_k):
            # Importance sample seed
            z = zLs[sample]

            # Path representations and distributions
            zs = [z]
            qs = [QzLGx]

            # Send z_l through all except last downward-pass block
            for i in range(self.layers - 1):
                # Decoder parameters
                mu_p, var_p = self.decoder[i](z)

                # Save first set of decoder parameters (deterministic)
                if i == 0:
                    mu_p_Lm1 += [mu_p]
                    var_p_Lm1 += [var_p]

                # Precision-weighted parameters
                mu_q, sd_q = self.precision_weighted(mu_q_hats[i + 1], var_q_hats[i + 1], mu_p, var_p)

                # Sample and make importance sample save
                dist = Normal(mu_q, sd_q + 1e-6)
                z = dist.rsample()

                zs += [z]
                qs += [dist]

            # add to importance sampling data collection
            imp_qs += [qs]
            imp_zs += [zs]

        return imp_zs, imp_qs, mu_p_Lm1, var_p_Lm1

    def decode(self, imp_zs, mu_p_Lm1, var_p_Lm1, imp_k):
        """ Part 1: Importance sampling data collection """
        # Importance weighted data: List of lists where each inner list describes the data for an importance sampling
        # path from z_1 to z_L.
        imp_first_dists = []
        imp_last_dist = []

        """ Part 2: Data through network """
        # Send each importance sample through decoder
        for sample in range(imp_k):
            # Saved deterministic parameters found during stochastic downward-pass in encoder are reused in the decoder
            # to get the identical first decoder distribution
            dist = Normal(mu_p_Lm1[sample], var_p_Lm1[sample] ** 0.5 + 1e-6)
            first_dists = [dist]

            # Get all decoder latent representations for importance sample
            zs = imp_zs[sample]

            # Send latent representations from the encoder through all but the last decoder block
            for (block, z) in zip(self.decoder[1:-1], zs[1:-1]):
                # Parameters from decoder
                mu_p, var_p = block(z)

                # Define distribution and save
                dist = Normal(mu_p, var_p ** 0.5 +1e-6)
                first_dists += [dist]

            # Send signal through the last decoder block
            logits = self.decoder[-1](zs[-1])

            # Define distribution
            last_dist = Bernoulli(logits=logits)

            # Add to importance sampling data collection (global save)
            imp_first_dists += [first_dists]
            imp_last_dist += [last_dist]

        return imp_first_dists, imp_last_dist

    def forward(self, x, imp_k):
        # Save distributions
        PzL = [self.prior(batch_size=x.size(0))]

        # Run enccoder blocks (with importance sampling)
        imp_zs, imp_qs, mu_p_Lm1, var_p_Lm1 = self.encode(x, imp_k)

        # Distributions from decoder
        imp_first_ps, imp_last_p = self.decode(imp_zs, mu_p_Lm1, var_p_Lm1, imp_k)

        # For each important sample save the prior distribution along with the other decoder distributions (except
        # the last decoder distribution q(x|z_1) which is saved separately). The goal with this is to be able to
        # calculate the loss easily.
        imp_other_ps = []

        for dists in imp_first_ps:
            temp = PzL + dists
            imp_other_ps += [temp]

        return imp_zs, imp_other_ps, imp_qs, imp_last_p

    def VariationalInference(self, x, imp_k, warm_up, epoch):
        # Forward pass through the model
        #imp_zs, imp_other_ps, imp_qs, imp_last_p = self.forward(x, imp_k)
        imp_zs, imp_other_ps, imp_qs, imp_last_p = self.forward(x, imp_k)
        imp_zs = [i[::-1] for i in imp_zs]

        # Current beta value (linear interpolation up to last 'warm-up' epoch)
        if warm_up:
            beta = min(1.0 / warm_up * epoch, 1.0)
        else:
            beta = 1.0

        # Calculate estimate (importance weighted estimate not possible) of the reconstruction loss (for later
        # comparison with the DGD models). Here one importance path used.
        rec_loss = float('inf')

        # Calculate for each importance sample path
        batch_size = imp_zs[0][0].size(0)
        imp_elbo = torch.zeros(imp_k, batch_size).to(self.device)
        imp_beta_elbo = torch.zeros(imp_k, batch_size).to(self.device)

        for k in range(imp_k):
            """ Caluclate probabilities for events from all encoder and decoder distributions (using log rules) """
            # In case of 3 layers: [Pz3(z3), Pz2Gz3(z2), Pz1Gz2(z1)]
            log_probs1 = [imp_other_ps[k][i].log_prob(imp_zs[k][::-1][i]).sum(dim=1) for i in range(self.layers)]

            # In case of 3 layers: [Qz1Gx(z1), Qz2Gz1(z2), Qz3Gz2(z3)]
            log_probs2 = [imp_qs[k][i].log_prob(imp_zs[k][::-1][i]).sum(dim=1) for i in range(self.layers)]

            # In case of any number of layers: [PxGz1(x)]
            log_probs3 = imp_last_p[k].log_prob(x).sum(dim=1)

            """ Caluclate final probabilities by combining the probabilities from above (using log rules)  """
            # In case of 3 layers: log_Pz = log_Pz3 + log_Pz2Gz3 + log_Pz1Gz2
            log_Pz = sum(log_probs1)

            # In case of 3 layers: log_QzGx = log_Qz1Gx + log_Qz2Gz1 + log_Qz3Gz2
            log_QzGx = sum(log_probs2)

            # In case of any number of layers: log_PxGz = log_PxGz1
            log_PxGz = log_probs3

            # Compute ELBO and beta-ELBO
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
        # Define and sample the prior distribution p(z_L)
        PzL = self.prior(batch_size=batch_size)
        z = PzL.rsample()

        # Define the observation model p(x|z) = B(x | g(z)). Notice I need not wait with sending the signal through the
        # last decoder block as I don't need to save the distributions in a special way for a loss function (as in the
        # encoder function).

        for i in range(len(self.decoder) - 1):
            # Get p(x|z) for current x and z
            mu_p, var_p = self.decoder[i](z)

            # Define distribution and sample
            PxGz = Normal(mu_p, var_p ** 0.5)
            z = PxGz.rsample()

        logits = self.decoder[-1](z)
        dist = Bernoulli(logits=logits)

        return {'PxGz1': dist, 'PzL': PzL, 'z': z}

    def precision_weighted(self, mu_q_hat, var_q_hat, mu_p, var_p):
        # Use formulae from LVAE paper
        sd_q = torch.ones(mu_q_hat.shape).to(self.device) / (var_q_hat ** (-1) + var_p ** (-1))
        mu_q = sd_q * 2 * (mu_q_hat * var_q_hat ** (-1) + mu_p * var_p ** (-1))

        return mu_q, sd_q
