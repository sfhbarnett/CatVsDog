import CatVsDog
import os
import torch
import Net
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

num_epochs = 15

#mainpath = r'C:\Users\MBISFHB\Documents\code\Python Scripts\Dogsvscats'
mainpath = '/Users/samuelbarnett/Documents/DL_data/'

trainpath = mainpath+r'train'
testpath = mainpath+r'test1'

trainset = os.listdir(trainpath)
dataset = CatVsDog.CatVsDog(trainset,trainpath,transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))

trainloader = torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True,num_workers=0)


net = Net.Net()
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
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
torch.save(net,mainpath+'\savedmodel.pt')
print('Model saved')
