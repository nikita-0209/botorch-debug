import torch

# from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam
from torch.nn import Linear
from torch.nn import MSELoss
from torch.nn import Sequential, ReLU, Dropout
from torch import tensor
import numpy as np
from abc import ABC
import gpytorch



"""
Tutorial: https://botorch.org/tutorials/compare_mc_analytic_acquisition
"""

neg_hartmann6 = Hartmann(dim=6, negate=True)

train_x = torch.rand(10, 6)
train_y = neg_hartmann6(train_x).unsqueeze(-1)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # batch_shape=torch.Size([config.BATCH])
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
            # batch_shape=torch.Size([config.BATCH])
        )

    def forward(self, x):
        mean_x = self.mean_module(x) #101
        covar_x = self.covar_module(x) #(train+test, train+test)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = ExactGPModel(train_x, train_y, likelihood)

gp.train()
likelihood.train()

from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import add_output_dim
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood


class myGPModel(SingleTaskGP):
    def __init__(self, gp, trainX, trainY):
        super().__init__(trainX, trainY)
        self.model = gp
    
    @property
    def num_outputs(self) -> int:
        return super().num_outputs

    @property
    def batch_shape(self):
        """
        This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.

        """
        return super().batch_shape

    def posterior(self, X, output_indices = None, observation_noise= False, posterior_transform= None):
        """
        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.
        Returns:
            A `GPyTorchPosterior` object, representing `batch_shape` joint
            distributions over `q` points and the outputs selected by
            `output_indices` each. Includes observation noise if specified.
        """
        self.eval()  # make sure model is in eval mode
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)
        # with gpt_posterior_settings():
            # insert a dimension for the output dimension
        if self._num_outputs > 1:
            X, output_dim_idx = add_output_dim(
                X=X, original_batch_shape=self._input_batch_shape
            )
        mvn = self(X)
        # if observation_noise is not False:
        #     if torch.is_tensor(observation_noise):
        #         # TODO: Validate noise shape
        #         # make observation_noise `batch_shape x q x n`
        #         obs_noise = observation_noise.transpose(-1, -2)
        #         mvn = self.likelihood(mvn, X, noise=obs_noise)
        #     elif isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
        #         # Use the mean of the previous noise values (TODO: be smarter here).
        #         noise = self.likelihood.noise.mean().expand(X.shape[:-1])
        #         mvn = self.likelihood(mvn, X, noise=noise)
        #     else:
        #         mvn = self.likelihood(mvn, X)
        if self._num_outputs > 1:
            mean_x = mvn.mean
            covar_x = mvn.lazy_covariance_matrix
            output_indices = output_indices or range(self._num_outputs)
            mvns = [
                MultivariateNormal(
                    mean_x.select(dim=output_dim_idx, index=t),
                    covar_x[(slice(None),) * output_dim_idx + (t,)],
                )
                for t in output_indices
            ]
            mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
        posterior = GPyTorchPosterior(mvn=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
proxy = myGPModel(gp, train_x, train_y)

qMES = qMaxValueEntropy(proxy, candidate_set = train_x, num_fantasies=1, use_gumbel=True)
test_X=torch.tensor([[[0.8754, 0.9025, 0.5862, 0.1580, 0.3266, 0.7930]],

        [[0.1407, 0.2835, 0.0574, 0.7165, 0.2836, 0.8033]],

        [[0.1043, 0.4672, 0.7695, 0.5995, 0.2715, 0.7897]],

        [[0.6130, 0.8399, 0.3882, 0.2005, 0.5959, 0.5445]],

        [[0.5849, 0.9051, 0.8367, 0.1182, 0.3853, 0.9588]],

        [[0.4114, 0.7935, 0.0299, 0.3348, 0.1985, 0.3097]],

        [[0.0172, 0.8890, 0.6926, 0.1963, 0.3057, 0.2855]],

        [[0.6131, 0.9267, 0.6613, 0.1429, 0.3706, 0.3486]],

        [[0.5914, 0.8657, 0.4393, 0.6715, 0.7866, 0.7446]],

        [[0.6269, 0.9950, 0.0640, 0.4415, 0.1140, 0.6024]]])
 
#because in the forward call, qMES adds a dimension but it also 
# does not accept textX in shape b x 1 xd
# with torch.no_grad():
#     mes = qMES(test_X)
# print(mes)

for num in range(10000):
    test_x = torch.rand(10, 6)
    # test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_X)
        mes_arr = mes.detach().numpy()
        verdict = np.all(mes_arr>0)
    if not verdict:
        print(mes)
