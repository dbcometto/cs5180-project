"""Some testing with pytorch"""

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

# # 1. Gradient of loss is stored in leaf tensors
# x = torch.tensor([1.0,2.0,3.0])
# W = torch.randn(3,requires_grad=True)

# y = x @ W

# loss = y**2
# loss.backward()

# print(W.grad)



# 2. NN making
torch.manual_seed(2025)
X = torch.randn(100,2)
# print(f"X: {X[:2,:2]}")
# print(X.shape)

def fn(X: torch.Tensor):
    return torch.stack([
    X[:,0] + X[:,1],
    torch.sin(2*np.pi*(X[:,0] - X[:,1]))
],dim=1)

Y = fn(X)
# print(f"Y: {Y[:2,:2]}")
# print(Y.shape)

model = torch.nn.Linear(2,2)

# print(model.weight)
# print(model.bias)


def loss_fn(pred: torch.Tensor, truth: torch.Tensor):
    return torch.mean((pred-truth)**2)


alpha = 0.1
episodes = 1000
for i in tqdm(range(episodes)):
    Y_pred = model(X)
    # print(f"Ypred: {Y_pred[:2,:2]}")
    # print(Y_pred.shape)

    # L = torch.mean((Y_pred-Y)**2)
    # print(L.shape)
    # print(f"Loss: {L}")
    

    loss = loss_fn(Y_pred,Y)
    tqdm.write(f"Loss: {loss}")

    loss.backward()
    # print(model.weight.grad)
    # print(model.bias.grad)


    with torch.no_grad():
        model.weight -= alpha*model.weight.grad
        model.bias -= alpha*model.bias.grad
    model.zero_grad()



xvals = torch.linspace(-5,5,10)
yvals = torch.linspace(-5,5,10)

Xgrid,Ygrid = torch.meshgrid(xvals,yvals,indexing='ij')
print(Xgrid.shape)

xyvals = torch.stack([Xgrid,Ygrid],dim=-1)
print(xyvals.shape)
xyvals_flat = xyvals.reshape((100,2))

zvals_flat = fn(xyvals_flat)
zvals = zvals_flat.reshape(10,10,2)
print(zvals.shape)


predvals_flat = model(xyvals_flat).detach().numpy()
predvals = predvals_flat.reshape(10,10,2)

fig,axs = plt.subplots(2,2,figsize=(12,12))

axs[0,0].imshow(zvals[:,:,0],'YlGn')
axs[1,0].imshow(predvals[:,:,0],'YlGn')

axs[0,1].imshow(zvals[:,:,1],'YlGn')
axs[1,1].imshow(predvals[:,:,1],'YlGn')


plt.show()
