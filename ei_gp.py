import torch

# from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam


"""
Tutorial: https://botorch.org/tutorials/compare_mc_analytic_acquisition
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

"Epoch  10/10 - Loss: 1.838 "
from botorch.acquisition import ExpectedImprovement

model.eval()

best_value = train_y.max()
EI = ExpectedImprovement(model=model, best_f=best_value)

test_x=torch.tensor([[[0.8754, 0.9025, 0.5862, 0.1580, 0.3266, 0.7930]],

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
ei = EI(test_x)
print(ei)


ei = EI(test_x)
print(ei)

"""
EI: tensor([0.0058, 0.0065, 0.0078, 0.0056, 0.0068, 0.0064, 0.0074, 0.0064, 0.0048,
        0.0046], grad_fn=<MulBackward0>)
"""