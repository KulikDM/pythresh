import numpy as np
from tqdm import tqdm
import scipy.stats as stats
from sklearn.utils import check_array
import torch
from torch import nn
import torch.optim as opt
from torch.nn.functional import softplus
from torch.distributions import Normal, kl_divergence
from .base import BaseThresholder
from .thresh_utility import normalize, cut, gen_kde


class VAE(BaseThresholder):
    r"""VAE class for Variational AutoEncoder thresholder.

       Use a VAE to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the maximum minus the minimum of the
       reconstructed distribution probabilities after encoding.
       See :cite:`xiao2020vae` for details

       Parameters
       ----------

       verbose : bool, optional (default=False)
            display training progress

       device : str, optional (default='cpu')
            device for pytorch

       latent_dims : int, optional (default='auto')
            number of latent dimensions the encoder will map the scores to.
            Default 'auto' applies automatic dimensionality selection using
            a profile likelihood.

       random_state : int, optional (default=1234)
            random seed for the normal distribution. Can also be set to None

       epochs : int, optional (default=100)
            number of epochs to train the VAE

       batch_size : int, optional (default=64)
            batch size for the dataloader during training

       loss : str, optional (default='kl')
            Loss function during training

            - 'kl' : use the combined negative log likelihood and Kullback-Leibler divergence
            - 'mmd': use the combined negative log likelihood and maximum mean discrepancy

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

    """

    def __init__(self, verbose=False, device='cpu', latent_dims='auto',
                 random_state=1234, epochs=100, batch_size=64, loss='kl'):

        self.verbose = verbose
        self.device = device
        self.latent_dims = latent_dims
        self.dist = Normal
        self.epochs = epochs
        self.loss = loss
        self.batch_size = batch_size
        self.random_state = random_state
        if random_state:
            torch.manual_seed(random_state)

    def eval(self, decision):
        """Outlier/inlier evaluation process for decision scores.

        Parameters
        ----------
        decision : np.array or list of shape (n_samples)
                   which are the decision scores from a
                   outlier detection.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """

        decision = check_array(decision, ensure_2d=False)
        scores = normalize(decision.copy())

        if self.latent_dims=='auto':
            self.latent_dims = self._autodim(scores)

        decision = normalize(decision).astype(np.float32).reshape(-1,1)

        self.model = VAE_model(1, self.latent_dims,
                          self.random_state, self.dist,
                          self.loss).to(self.device)

        self.data = torch.utils.data.DataLoader(
                            torch.from_numpy(decision),
                            batch_size=self.batch_size,
                            shuffle=True)

        self._train()
        with torch.no_grad():
            z = self.model.reconstructed_probability(torch.from_numpy(decision).to(
                                        self.device)).to('cpu').detach().numpy()

        limit = np.max(z)-np.min(z)
        self.thresh_ = limit

        return cut(scores, limit)

    def _autodim(self, vals):
        ''' Estimate the latent dimension size using the method of Zhu and Ghodsi (2006) '''

        vals = np.sort(vals)[::-1]
        m = len(vals)
        profile_lik = []

        for i in range(1,m):

            mu1 = np.mean(vals[:i])
            mu2 = np.mean(vals[i:])
            sigma = np.sqrt((np.sum((vals[:i] - mu1) ** 2) +
                             np.sum((vals[i:] - mu2) ** 2)) / (m-2))

            profile_lik += [np.sum(stats.norm.logpdf(vals[:i], loc=mu1, scale=sigma)) +
                            np.sum(stats.norm.logpdf(vals[i:], loc=mu2, scale=sigma))]

        dims = round(np.log(np.argsort(profile_lik)[-1]))
        return dims

    def _train(self):

        optimizer = torch.optim.Adam(self.model.parameters(),
                                weight_decay=1e-4,
                                lr=1e-2)

        scheduler = opt.lr_scheduler.ExponentialLR(optimizer,
                                                  gamma=0.95)

        for epoch in (tqdm(range(self.epochs), ascii=True, desc="Training")
                        if self.verbose else range(self.epochs)):

            for x in self.data:

                x = x.to(self.device)
                optimizer.zero_grad()
                _ = self.model.forward(x)

                optimizer.step()

            scheduler.step()

class VAE_model(nn.Module):

    def __init__(self, input_size, latent_size,
                 random_state, dist, loss, L=16):

        super().__init__()
        self.L = L
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder = self.data_encoder(input_size, latent_size)
        self.decoder = self.data_decoder(latent_size, input_size)
        self.dist = dist
        self.loss = loss

        if random_state:
            torch.manual_seed(random_state)

        self.prior = self.dist(0, 1)

    def data_encoder(self, input_size, latent_size):

        return nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, latent_size * 2)
        )

    def data_decoder(self, latent_size, output_size):

        return nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_size * 2)
        )

    def forward(self, x):

        pred_result = self.predict(x)
        x = x.unsqueeze(0)

        # average over sample dimension
        log_lik = self.dist(pred_result['recon_mu'],
                            pred_result['recon_sigma']).log_prob(x).mean(
                            dim=0)
        log_lik = log_lik.mean(dim=0).sum()

        # calculate the kl divergence and the forward loss
        if self.loss=='kl':
            kl = kl_divergence(pred_result['latent_dist'],
                            self.prior).mean(dim=0).sum()
            loss = kl - log_lik

            return dict(loss=loss, kl=kl, recon_loss=log_lik, **pred_result)

        # calculate the mmd and the forward loss
        else:

            x = x.squeeze()
            batch_size = len(x)

            z = pred_result['latent_dist'].rsample([self.L])
            z = z.view(self.L * batch_size, self.latent_size)

            x = self.prior.rsample([self.L * batch_size * self.latent_size])
            x = x.view(self.L * batch_size, self.latent_size)

            mmd = self.compute_mmd(z, x)

            loss = mmd - log_lik

            return dict(loss=loss, kl=mmd, recon_loss=log_lik, **pred_result)

    def predict(self, x):

        batch_size = len(x)

        latent_mu, latent_sigma = self.encoder(x).chunk(2, dim=1)
        latent_sigma = softplus(latent_sigma)

        dist = self.dist(latent_mu, latent_sigma)
        z = dist.rsample([self.L])
        z = z.view(self.L * batch_size, self.latent_size)

        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_mu = recon_mu.view(self.L, *x.shape)

        recon_sigma = softplus(recon_sigma)
        recon_sigma = recon_sigma.view(self.L, *x.shape)

        return dict(latent_dist=dist, latent_mu=latent_mu,
                    latent_sigma=latent_sigma, recon_mu=recon_mu,
                    recon_sigma=recon_sigma, z=z)

    def reconstructed_probability(self, x):

        with torch.no_grad():
            pred = self.predict(x)

        recon_dist = self.dist(pred['recon_mu'], pred['recon_sigma'])
        x = x.unsqueeze(0)
        p = recon_dist.log_prob(x).exp().mean(dim=0).mean(dim=-1)

        return p

    def compute_kernel(self, x, y):

        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        x = x.view(x_size, 1, dim)
        y = y.view(1, y_size, dim)
        tx = x.expand(x_size, y_size, dim)
        ty = y.expand(x_size, y_size, dim)

        kernel_input = (tx - ty).pow(2).mean(2)/float(dim)

        return torch.exp(-kernel_input)

    def compute_mmd(self, x, y):

        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)

        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()

        return mmd


