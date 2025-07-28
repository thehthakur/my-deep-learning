import torch
import torch.nn.functional as F

y: torch.Tensor = torch.tensor(data=[1.0])

x1: torch.Tensor = torch.tensor(data=[1.1])
w1: torch.Tensor = torch.tensor(data=[2.2], requires_grad=True)
b: torch.Tensor = torch.tensor(data=[0.0], requires_grad=True)

z: torch.Tensor = (w1 @ x1) + b
a: torch.Tensor = torch.sigmoid(z)

loss: torch.Tensor = F.binary_cross_entropy(input=a, target=y)

loss.backward()
print(w1.grad)
print(b.grad)
