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


"""
Tutorial: https://botorch.org/tutorials/compare_mc_analytic_acquisition
        https://botorch.org/tutorials/compare_mc_analytic_acquisition
"""


"""
Use this trianing data as easy to compute the maximum
"""

neg_hartmann6 = Hartmann(dim=6, negate=True)

train_x = torch.rand(10, 6)
train_y = neg_hartmann6(train_x).unsqueeze(-1)
mlp = Sequential(Linear(6, 128), ReLU(), Dropout(0.1), Linear(128, 128), Dropout(0.1), ReLU(), Linear(128, 1))

NUM_EPOCHS = 10

mlp.train()
optimizer = Adam(mlp.parameters())
criterion = MSELoss()


for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output = mlp(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
         )
    optimizer.step()

"""
Epoch  10/10 - Loss: 0.012 
"""

from botorch.acquisition import ExpectedImprovement
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior



class NN_Model(Model):
    def __init__(self, nn):
        super().__init__()
        self.model = nn
        self._num_outputs = 1
        self.nb_samples = 20

    def posterior(self, X, observation_noise = False, posterior_transform = None):
        # called during forward call, X.shape = (10, 1, 6)
        super().posterior(X, observation_noise, posterior_transform)
        # X.shape = 10x1x6--> b x q x d
        self.model.train(mode=True)
        with torch.no_grad():
             outputs = torch.hstack([self.model(X) for _ in range(self.nb_samples)])
        mean = torch.mean(outputs, axis=1) #10, 1
        var = torch.var(outputs, axis=1) #10, 1
        covar = [torch.diag(var[i]) for i in range(X.shape[0])]
        covar = torch.stack(covar, axis = 0)
        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)
        return posterior

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        """
        This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.

        """
        return torch.Size([10, 1, 1])
    
proxy = NN_Model(mlp)

from botorch.acquisition import qExpectedImprovement, ExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler

best_value = train_y.max()
sampler = SobolQMCNormalSampler(num_samples=500, seed=0, resample=False)


test_x = tensor([[[0.8754, 0.9025, 0.5862, 0.1580, 0.3266, 0.7930]],

        [[0.1407, 0.2835, 0.0574, 0.7165, 0.2836, 0.8033]],

        [[0.1043, 0.4672, 0.7695, 0.5995, 0.2715, 0.7897]],

        [[0.6130, 0.8399, 0.3882, 0.2005, 0.5959, 0.5445]],

        [[0.5849, 0.9051, 0.8367, 0.1182, 0.3853, 0.9588]],

        [[0.4114, 0.7935, 0.0299, 0.3348, 0.1985, 0.3097]],

        [[0.0172, 0.8890, 0.6926, 0.1963, 0.3057, 0.2855]],

        [[0.6131, 0.9267, 0.6613, 0.1429, 0.3706, 0.3486]],

        [[0.5914, 0.8657, 0.4393, 0.6715, 0.7866, 0.7446]],

        [[0.6269, 0.9950, 0.0640, 0.4415, 0.1140, 0.6024]]])
# Expected X to be `batch_shape x q=1 x d`, but got X with shape torch.Size([10, 6]).
# test_x = test_x.unsqueeze(-2)



MC_EI = qExpectedImprovement(
    proxy, best_f=best_value, sampler=sampler
)
mc_ei = MC_EI(test_x)
print(mc_ei)

torch.manual_seed(seed=0)
EI = ExpectedImprovement(model=proxy, best_f=best_value)
ei = EI(test_x)
print(ei)


print(torch.norm(mc_ei - ei))


# Validated as values are super similar



