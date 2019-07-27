import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
  
  def __init__(self,):
    super(LeNet, self).__init__()
    
    self.cnn_model = nn.Sequential(
    nn.Conv2d(3,6,5),
        nn.ReLU(),
        nn.AvgPool2d(2, stride= 2),
        nn.Conv2d(6, 16, 5,),
        nn.ReLU(),
        nn.AvgPool2d(2, stride = 2),
        
    )
    
    self.fc_model = nn.Sequential(
        
        nn.Linear(400,120),
        nn.ReLU(),
        nn.Linear(120,84),
        nn.ReLU(),
        nn.Linear(84,10)
    )
    
    
  def forward(self,x):
    
#     print(x.shape)
    x = self.cnn_model(x)
#     print(x.shape)
    x = x.view(x.shape[0], -1)
#     print(x.shape)
    x = self.fc_model(x)
#     print(x.shape)
    return x
  

def evaluation(dataloader, net):
  
  total, correct = 0,0
  for data in dataloader:
    inputs, labels = data
    outputs = net(inputs)
    _, pred = torch.max(outputs, 1)
    total +=  labels.size(0)
    correct += (labels == pred).sum().item()
    
  return (100 * correct)/(total)
  
  
  batch_size = 128
trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform =  transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)

testset = torchvision.datasets.CIFAR10(root = '.data', train = False, download = True, transform = transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False)



  
  
net5 = LeNet()
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(net5.parameters())


loss_arr = []
loss_epoch_arr = []
max_epoches = 16

for epoch in range(max_epoches):
  
  for i, data in enumerate(trainloader,0):
    
    image,label = data
    
    opt.zero_grad()
    
    output = net5.forward(image)
    
    loss = loss_fn(output, label)
    
    loss.backward()
      
    opt.step()
    
    loss_arr.append(loss.item())
    
  loss_epoch_arr.append(loss.item())
  print('epoches {}/{}, test_acc = {}, train= {}'.format(epoch, max_epoches, evaluation(trainloader, net5), evaluation(testloader, net5)))
