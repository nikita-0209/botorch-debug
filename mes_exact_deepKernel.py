import math
# import tqdm
import torch
import gpytorch
from tqdm.notebook import tqdm

import urllib.request
import os
from scipy.io import loadmat
from math import floor
from botorch.test_functions import Hartmann


# this is for running the notebook in our testing framework
smoke_test = True
neg_hartmann6 = Hartmann(dim=6, negate=True)

train_x = torch.rand(10, 6)
train_y = neg_hartmann6(train_x)

# define a testloader similar to how the trainloader is defined
test_x = torch.rand(10, 6)
test_y = neg_hartmann6(test_x)
# bring train_x, train_y, test_x, test_y to cuda
train_x = train_x.cuda()
train_y = train_y.cuda()
test_x = test_x.cuda()
test_y = test_y.cuda()

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

data_dim = train_x.size(-1)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1024))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1024, 512))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(512, 256))
        # self.add_module('relu3', torch.nn.ReLU())
        # self.add_module('linear4', torch.nn.Linear(50, 50))

feature_extractor = LargeFeatureExtractor()

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel()
        )
            self.feature_extractor = feature_extractor
            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            projected_x = self.feature_extractor(x) #projected_x = (1600, 2)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

training_iterations = 10

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

def train():
    iterator = tqdm(range(training_iterations))
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()

train()


from botorch.models import SingleTaskGP
from gpytorch.distributions import MultivariateNormal
from botorch.posteriors import GPyTorchPosterior

class myGPModel(SingleTaskGP):
    def __init__(self, gp, trainX, trainY):
        super().__init__(trainX, trainY)
        self.model = gp
    
    @property
    def num_outputs(self) -> int:
        return super().num_outputs

    @property
    def batch_shape(self):
       return super().batch_shape

    def posterior(self, X, output_indices = None, observation_noise= False, posterior_transform= None):
        self.eval()  # make sure model is in eval mode
        X = X.to('cuda')
        mvn = self.model(X)
        posterior = GPyTorchPosterior(mvn=mvn)
        return posterior

from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
proxy = myGPModel(model, train_x, train_y.unsqueeze(-1))
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

import numpy as np
for num in range(10000):
    test_x = torch.rand(10, 6)
    # test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_X.to('cuda'))
        mes_arr = mes.cpu().detach().numpy()
        verdict = np.all(mes_arr>0)
    if not verdict:
        print(mes)


# model.eval()
# likelihood.eval()
# with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
#     preds = model(test_x)

# print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))