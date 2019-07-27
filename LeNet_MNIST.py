class LeNet_mnist(nn.Module):
  
  def __init__(self,):
    super(LeNet_mnist, self).__init__()
    
    self.cnn_model = nn.Sequential(
    nn.Conv2d(1,6,5),
        nn.ReLU(),
        nn.AvgPool2d(2, stride= 2),
        nn.Conv2d(6, 16, 5,),
        nn.ReLU(),
        nn.AvgPool2d(2, stride = 2),
        
    )
    
    self.fc_model = nn.Sequential(
        
        nn.Linear(256,120),
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
  
  
  
  
batch_size = 128
trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform =  transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)

testset = torchvision.datasets.MNIST(root = '.data', train = False, download = True, transform = transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False)


net_mnist = LeNet_mnist().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(net_mnist.parameters(), weight_decay=0.001)





loss_epoch_arr = []
max_epoch = 50

for i in range(max_epoch):
  print(i)
  
  for j, data in enumerate(trainloader,0):
    
    input_img, label = data
    
    input_img, label = input_img.to(device), label.to(device)
    
    opt.zero_grad()
    
    output = net_mnist.forward(input_img)
    
    loss = loss_fn(output, label)
    
    loss.backward()
    
    opt.step()
   
  print('epoches {}/{}, train_acc = {}, test= {}'.format(i, max_epoches, evaluation(trainloader, net_mnist), evaluation(testloader, net_mnist)))
    
    


