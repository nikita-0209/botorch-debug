from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch import nn
import torch
import os
import gpytorch
import math
from tqdm.notebook import tqdm
from botorch.test_functions import Hartmann
from torch.nn import Sequential, ReLU, Dropout, Linear, MSELoss
from torch.utils.data import DataLoader, Dataset

"""
Tutorial: https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html#Set-up-data-augmentation
"""

neg_hartmann6 = Hartmann(dim=6, negate=True)

class Data(Dataset):
    def __init__(self, trainX, trainY) -> None:
        super().__init__()
        self.trainX = trainX
        self.trainY = trainY
    def __getitem__(self, index):
        return self.trainX[index], self.trainY[index]
    def __len__(self):
        return len(self.trainX)

    
train_x = torch.rand(10, 6)
train_y = neg_hartmann6(train_x).unsqueeze(-1)
dataset = Data(train_x, train_y)
train_loader = DataLoader(dataset, batch_size=1)

test_x = torch.rand(10, 6)
test_y = neg_hartmann6(test_x).unsqueeze(-1)
test_dataset = Data(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=1)

"""
If we are doing DropoutRegressor technique, then it must have a dropout layer.
Removed the last linear layer because GP operates on the features of the MLP
"""
mlp = Sequential(Linear(6, 32), ReLU(), Dropout(0.5), Linear(32, 32), Dropout(0.5), ReLU())

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, GridInterpolationVariationalStrategy
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel

class GaussianProcessLayer(ApproximateGP):
    def __init__(self, num_dim, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=32, batch_shape=torch.Size([num_dim]))
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                            VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True),
                            num_tasks=num_dim
                        )
        super(GaussianProcessLayer, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

inducing_points = train_x[:5, :]
model = GaussianProcessLayer(num_dim = 32, inducing_points=inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, inducing_points=inducing_points)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res

model = DKLModel(mlp, num_dim=32)
likelihood = gpytorch.likelihoods.GaussianLikelihood()


# If you run this example without CUDA, I hope you like waiting!
if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()


n_epochs = 20
lr = 0.1
optimizer = SGD([
    {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_x))


def train(epoch):
    model.train()
    likelihood.train()

    minibatch_iter = tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    with gpytorch.settings.num_likelihood_samples(8):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = -mll(output, target)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())

# def test():
#     model.eval()
#     likelihood.eval()
#     means = torch.tensor([0.])
#     with torch.no_grad():
#         for x_batch, y_batch in test_loader:
#             x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
#             preds = model(x_batch)
#             mean_preds = preds.mean.cpu()
#             means = torch.cat([means, mean_preds])
#     means = means[1:]
#     print('Test MAE: {}'.format(torch.mean(torch.abs(means - test_y.cpu()))))

for epoch in range(1, n_epochs + 1):
    with gpytorch.settings.use_toeplitz(False):
        train(epoch)
        # test()
    scheduler.step()
    state_dict = model.state_dict()
    likelihood_state_dict = likelihood.state_dict()

# from botorch.acquisition import qMaxValueEntropy
# qMES = qMaxValueEntropy(model, train_x, num_fantasies=1)
# test_X=torch.tensor([[[0.8754, 0.9025, 0.5862, 0.1580, 0.3266, 0.7930]],

#         [[0.1407, 0.2835, 0.0574, 0.7165, 0.2836, 0.8033]],

#         [[0.1043, 0.4672, 0.7695, 0.5995, 0.2715, 0.7897]],

#         [[0.6130, 0.8399, 0.3882, 0.2005, 0.5959, 0.5445]],

#         [[0.5849, 0.9051, 0.8367, 0.1182, 0.3853, 0.9588]],

#         [[0.4114, 0.7935, 0.0299, 0.3348, 0.1985, 0.3097]],

#         [[0.0172, 0.8890, 0.6926, 0.1963, 0.3057, 0.2855]],

#         [[0.6131, 0.9267, 0.6613, 0.1429, 0.3706, 0.3486]],

#         [[0.5914, 0.8657, 0.4393, 0.6715, 0.7866, 0.7446]],

#         [[0.6269, 0.9950, 0.0640, 0.4415, 0.1140, 0.6024]]])
# with torch.no_grad():
#     mes = qMES(test_X)
# print(mes)


# from gpytorch.distributions import MultivariateNormal
# from torch import distributions
# from botorch.posteriors import GPyTorchPosterior
# from botorch.models.model import Model

# class NN_Model(Model):
#     def __init__(self, nn):
#         super().__init__()
#         self.model = nn
#         self._num_outputs = 1
#         self.nb_samples = 20
#         """
#         train_inputs: A `n_train x d` Tensor that the model has been fitted on.
#                 Not required if the model is an instance of a GPyTorch ExactGP model.
#         """

#     def posterior(self, X, observation_noise = False, posterior_transform = None):
#         super().posterior(X, observation_noise, posterior_transform)
#         self.model.train()
#         with torch.no_grad():
#              outputs = torch.hstack([self.model(X) for _ in range(self.nb_samples)])
#         mean = torch.mean(outputs, axis=1)
#         var = torch.var(outputs, axis=1)
#         if len(X.shape)==2:
#             covar = torch.diag(var)
#         elif len(X.shape)==4:
#             var_element = var[0]
#             covar = [torch.diag(var[i][0]) for i in range(X.shape[0])]
#             covar = torch.stack(covar, axis = 0)
#             covar = covar.unsqueeze(-1)
#         mvn = MultivariateNormal(mean, covar)
#         tmvn = distributions.MultivariateNormal(mean, covar)
#         posterior = GPyTorchPosterior(mvn)
#         return posterior

#     @property
#     def num_outputs(self) -> int:
#         return self._num_outputs

#     @property
#     def batch_shape(self):
#         """
#         This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
#         to the `posterior` method returns a Posterior object over an output of
#         shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.

#         """
#         return torch.Size([])


