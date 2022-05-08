import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
# a = np.array([[1,2],[3,4]])
# print(a)
# rho = torch.FloatTensor([1 for _ in range(14)]).unsqueeze(0)
# rho = rho - torch.max(rho)
# print(rho)
# print(F.softmax(rho,dim=1))
# rho.view(-1,2)
# print(rho.view(-1,2))


x = torch.randn(5,3,4)
y = torch.randn(5,3,1)
# print(x)
# print(y)
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=2,
    shuffle=True


)
for step, (batch_x, batch_y) in enumerate(loader):
    print(batch_y)
    print(batch_y.view(-1,))

