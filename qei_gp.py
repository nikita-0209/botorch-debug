import torch

# from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam


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
model = SingleTaskGP(train_X=train_x, train_Y=train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
mll = mll.to(train_x)

NUM_EPOCHS = 10

model.train()
optimizer = Adam([{'params': model.parameters()}], lr=0.1)


for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output = model(train_x)
    loss = - mll(output, model.train_targets)
    loss.backward()
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} " 
         )
    optimizer.step()

"Epoch  10/10 - Loss: 1.777 "

from botorch.acquisition import qExpectedImprovement, ExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler

best_value = train_y.max()
sampler = SobolQMCNormalSampler(num_samples=500, seed=0, resample=False)        
MC_EI = qExpectedImprovement(
    model, best_f=best_value, sampler=sampler
)


EI = ExpectedImprovement(model=model, best_f=best_value)

test_x = torch.rand(10, 6)
# Expected X to be `batch_shape x q=1 x d`, but got X with shape torch.Size([10, 6]).
test_x = test_x.unsqueeze(-2)

mc_ei = MC_EI(test_x)
print(mc_ei)

"""
tensor([0.0230, 0.0217, 0.0229, 0.0222, 0.0228, 0.0223, 0.0237, 0.0230, 0.0198,
        0.0218], grad_fn=<MeanBackward1>)
"""

ei = EI(test_x)
print(ei)

"""
tensor([0.0233, 0.0219, 0.0231, 0.0225, 0.0230, 0.0225, 0.0240, 0.0232, 0.0200,
        0.0220], grad_fn=<MulBackward0>)
"""

print(torch.norm(mc_ei - ei))
# tensor(0.0007, grad_fn=<CopyBackwards>)


# Validated as values are super similar



