"""Some testing with pytorch"""

import torch
from tqdm import tqdm
from collections import OrderedDict 

import matplotlib.pyplot as plt
import numpy as np

# # 1. Gradient of loss is stored in leaf tensors
# x = torch.tensor([1.0,2.0,3.0])
# W = torch.randn(3,requires_grad=True)

# y = x @ W

# loss = y**2
# loss.backward()

# print(W.grad)



class MyNet(torch.nn.Module):
    """A network model"""

    def __init__(self):
        """Create my network model"""
        super().__init__()

        num = 64

        self.backbone = torch.nn.Sequential(
            OrderedDict(
                [
                    ("inputlayer", torch.nn.Linear(2,num)),
                    ("nonlinearlayer1", torch.nn.Tanh()),
                    ("linearlayer1", torch.nn.Linear(num,num)),
                    ("nonlinearlayer2", torch.nn.Tanh()),
                    ("linearlayer2", torch.nn.Linear(num,num)),
                    ("nonlinearlayer3", torch.nn.Tanh()),
                ]
            )
        )

        self.head = torch.nn.Linear(num,2)

    def forward(self,X):
        h = self.backbone(X)
        y = self.head(h)
        return y



# 2. NN making
torch.manual_seed(2025)
X = torch.randn(2000,2)*5
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

model = MyNet()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# print(model.weight)
# print(model.bias)


def loss_fn(pred: torch.Tensor, truth: torch.Tensor):
    return torch.mean((pred-truth)**2)

losses = []
testlosses = []

epochs = 50000
for i in tqdm(range(epochs)):
    
    optimizer.zero_grad()

    Y_pred = model(X)
    loss = loss_fn(Y_pred,Y)

    loss.backward()
    optimizer.step()
    
    losses.append(loss.detach().numpy())

    Xtest = torch.randn(2000,2)*5
    Y_pred = model(Xtest)
    testloss = loss_fn(fn(Xtest),Y_pred)
    testlosses.append(testloss)


    if i % (epochs/10) == 0:
        tqdm.write(f"Loss: {loss} | testloss: {testloss}")

    
    

# Training graph
fig,axs = plt.subplots(1,2,figsize=(12,6))  

axs[0].plot(losses,label="Loss")
axs[0].plot(testlosses,label="Test Loss")
axs[0].grid(True)
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[0].set_title("Loss vs Epoch")





# Testing graph


xvals = torch.linspace(-5,5,10)
yvals = torch.linspace(-5,5,10)

Xgrid,Ygrid = torch.meshgrid(xvals,yvals,indexing='ij')
print(Xgrid.shape)

xyvals = torch.stack([Xgrid,Ygrid],dim=-1)
print(xyvals.shape)
xyvals_flat = xyvals.reshape((100,2))

zvals_flat = fn(xyvals_flat)
zvals = zvals_flat.detach().numpy().reshape(10,10,2)
print(zvals.shape)


predvals_flat = model(xyvals_flat)
predvals = predvals_flat.detach().numpy().reshape(10,10,2)

test_loss = loss_fn(predvals_flat,zvals_flat)
print(f"Test Loss: {test_loss}")






fig,axs = plt.subplots(2,2,figsize=(12,12))

axs[0,0].imshow(zvals[:,:,0],'YlGn')
axs[1,0].imshow(predvals[:,:,0],'YlGn')

axs[0,1].imshow(zvals[:,:,1],'YlGn')
axs[1,1].imshow(predvals[:,:,1],'YlGn')


plt.show()
