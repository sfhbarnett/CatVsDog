import CatVsDog
import os
import torch
import Net
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

num_epochs = 30

mainpath = r'C:\Users\MBISFHB\Documents\code\Python Scripts\Dogsvscats' # Windows
#mainpath = '/Users/samuelbarnett/Documents/DL_data/' # Mac

trainpath = mainpath+r'\train'
testpath = mainpath+r'test1'
reportrate = 1000

trainset = os.listdir(trainpath)
imsize = 100
tforms = transforms.Compose([transforms.Resize((imsize,imsize)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = CatVsDog.CatVsDog(trainset,trainpath,transform=tforms,train=1)

trainloader = torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True,num_workers=0)

net = Net.Net(imsize)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        inputs = data['image']
        labels = data['annotation']
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % reportrate == reportrate-1:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / reportrate))
            running_loss = 0.0


print('Finished Training')
torch.save(net,mainpath+'\savedmodel.pt')
print('Model saved')
