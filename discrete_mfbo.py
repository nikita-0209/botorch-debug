from curses import keyname
import os
import torch


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

from botorch.test_functions.multi_fidelity import AugmentedHartmann


problem = AugmentedHartmann(negate=True).to(**tkwargs)
fidelities = torch.tensor([0.5, 0.75, 1.0], **tkwargs)

from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
# from botorch.fit import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity




def generate_initial_data(n=16):
    # generate training data
    train_x = torch.rand(n, 6, **tkwargs)
    # trainX.shape = (16, 6)
    train_f = fidelities[torch.randint(3, (n, 1))]
    # trainf.shape = (16, 1)
    train_x_full = torch.cat((train_x, train_f), dim=1) #(16, 7)
    # priblem is a synthetic test function used for getting trainY
    train_obj = problem(train_x_full).unsqueeze(-1)  # add output dimension
    # train_obj is basically trainY
    # 
    return train_x_full, train_obj


def initialize_model(train_x, train_obj):
    # define a surrogate model suited for a "training data"-like fidelity parameter
    # in dimension 6, as in [2]
    model = SingleTaskMultiFidelityGP(train_x, train_obj, outcome_transform=Standardize(m=1), data_fidelity=6)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model




bounds = torch.tensor([[0.0] * problem.dim, [1.0] * problem.dim], **tkwargs)
target_fidelities = {6: 1.0}

cost_model = AffineFidelityCostModel(fidelity_weights={6: 1.0}, fixed_cost=5.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)


def project(X, target_fidelities = None):
    if target_fidelities is None:
        target_fidelities = {-1: 1.0}
    # X.shape = torch.Size([128, 5, 1, 7])
    # the values of X chnage
    d = X.size(-1)
    # d = 7
    # normalize to positive indices
    # k = key, v = value 
    tfs = {k if k >= 0 else d + k: v for k, v in target_fidelities.items()}
    ones = torch.ones(*X.shape[:-1], device=X.device, dtype=X.dtype)
    # here we're looping through the feature dimension of X - this could be
    # slow for large `d`, we should optimize this for that case
    X_proj = torch.stack(
        [X[..., i] if i not in tfs else tfs[i] * ones for i in range(d)], dim=-1
    )
    return X_proj


    # return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

def get_mfkg(model):
    
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=7,
        columns=[6],
        values=[1],
    )
    
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=1024 if not SMOKE_TEST else 4,
        options={"batch_limit": 10, "maxiter": 200},
    )
        
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128 if not SMOKE_TEST else 2,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )

from botorch.optim.optimize import optimize_acqf_mixed


torch.set_printoptions(precision=3, sci_mode=False)

NUM_RESTARTS = 5 if not SMOKE_TEST else 2
RAW_SAMPLES = 128 if not SMOKE_TEST else 4
BATCH_SIZE = 4


def optimize_mfkg_and_get_observation(mfkg_acqf):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""

    # generate new candidates
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        fixed_features_list=[{6: 0.5}, {6: 0.75}, {6: 1.0}],
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        # batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x).unsqueeze(-1)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost


train_x, train_obj = generate_initial_data(n=16)

cumulative_cost = 0.0
N_ITER = 3 if not SMOKE_TEST else 1

for i in range(N_ITER):
    mll, model = initialize_model(train_x, train_obj)
    # fit_gpytorch_mll(mll)
    mfkg_acqf = get_mfkg(model)
    new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    cumulative_cost += cost