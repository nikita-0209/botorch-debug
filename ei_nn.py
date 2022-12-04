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
"""

neg_hartmann6 = Hartmann(dim=6, negate=True)

train_x = torch.rand(10, 6)
train_y = neg_hartmann6(train_x).unsqueeze(-1)

"""
If we are doing DropoutRegressor technique, then it must have a dropout layer
"""
mlp = Sequential(Linear(6, 1024), ReLU(), Dropout(0.5), Linear(1024, 1024), Dropout(0.5), ReLU(), Linear(1024, 1))

"""
For a very sparse network, with hidden_dim = 8, the expected improvement is very less even though the final loss of the mlp is similar.
EI dec as MLP dim dec
"""

NUM_EPOCHS = 10

mlp.train()
optimizer = Adam(mlp.parameters())
criterion = MSELoss()

seed = 123
torch.manual_seed(seed)

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

from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior
from torch.distributions import Normal

"""
To corss check that the implementation of posterior.mean and posterior.var is correct
"""

def calculate_EI(mean, sigma, best_f):
        u = (mean - best_f.expand_as(mean)) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei



class NN_Model(Model):
    def __init__(self, nn, best_value):
        super().__init__()
        self.model = nn
        self._num_outputs = 1
        self.nb_samples = 20
        self.best_value = best_value

    def posterior(self, X, observation_noise = False, posterior_transform = None):
        super().posterior(X, observation_noise, posterior_transform)
        self.model.train()
        with torch.no_grad():
             outputs = torch.hstack([self.model(X) for _ in range(self.nb_samples)])
        mean = torch.mean(outputs, axis=1)
        var = torch.var(outputs, axis=1)

        # mean = mean.squeeze(-1)
        # std = std.squeeze(-1)
        # covar = torch.diag(std)
        # covar = covar.unsqueeze(0)
        covar = [torch.diag(var[i]) for i in range(X.shape[0])]
        covar = torch.stack(covar, axis = 0)
        # torch.Size([10, 1, 1])
        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)

        print("Code calulated EI")
        ei = calculate_EI(mean, var.sqrt(), best_value)
        print(ei)
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

best_value = train_y.max() 
proxy = NN_Model(mlp, best_value)

EI = ExpectedImprovement(model=proxy, best_f=best_value)

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
mlp.train()
test_y = neg_hartmann6(test_x).unsqueeze(-1)
# tensor([[[0.0932]],

#         [[0.3219]],

#         [[0.9961]],

#         [[0.0926]],

#         [[0.1175]],

#         [[0.7353]],

#         [[0.1511]],

#         [[0.1324]],

#         [[0.0069]],

#         [[0.0659]]])
ei = EI(test_x)
print("EI Analytic Botorch:")
print(ei)

"""
Would be different naturally as now our surrogate model has a different loss
EI: tensor([0.0163, 0.0048, 0.0114, 0.0105, 0.0218, 0.0024, 0.0054, 0.0043, 0.0106,
        0.0048])

"""

"""
Spotted a negative value in EI computation
tensor([-9.2734e-09,  1.7604e-10,  2.2403e-08,  6.7140e-11,  6.8687e-08,
         9.7939e-15,  3.7211e-11,  2.5115e-09,  3.0061e-11,  1.5667e-14])
"""


from botorch.sampling import SobolQMCNormalSampler
sampler = SobolQMCNormalSampler(num_samples=500, seed=0, resample=False)
MC_EI = qExpectedImprovement(
    proxy, best_f=best_value, sampler=sampler
)
mc_ei = MC_EI(test_x)
print("qEI MC Botorch:")
print(mc_ei)

print(torch.norm(mc_ei - ei))


"""
Epoch  10/10 - Loss: 0.004 
tensor([0.0248, 0.0062, 0.0117, 0.0157, 0.0183, 0.0097, 0.0221, 0.0138, 0.0121,
        0.0118])
tensor([0.0315, 0.0022, 0.0100, 0.0119, 0.0143, 0.0095, 0.0161, 0.0218, 0.0173,
        0.0146])
tensor(0.0151)

Epoch  10/10 - Loss: 0.253 
tensor([5.9857e-13, 5.6269e-11, 4.0018e-18, 6.3921e-21, 2.2458e-13, 1.5842e-15,
        1.9121e-17, 3.7018e-17, 2.5472e-15, 2.2232e-14])
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
tensor(5.6272e-11)

Epoch  10/10 - Loss: 0.089 
tensor([0.0007, 0.0005, 0.0034, 0.0012, 0.0021, 0.0002, 0.0015, 0.0012, 0.0020,
        0.0013])
tensor([0.0016, 0.0006, 0.0013, 0.0006, 0.0008, 0.0003, 0.0015, 0.0010, 0.0048,
        0.0019])
tensor(0.0039)
"""


