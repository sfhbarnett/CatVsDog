import torch
import torchvision
import torchvision.transforms as transforms
import os


mainpath = 'C:\Users\MBISFHB\Documents\code\Python Scripts\Dogsvscats'

trainpath = mainpath+r'\train'
testpath = mainpath+r'\test1'

trainset = os.listdir(trainpath)



transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.VOCSegmentation(root='./data',year='2012', image_set='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testset = torchvision.datasets.VOCSegmentation(root='./data',year='2012', image_set='val', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)


