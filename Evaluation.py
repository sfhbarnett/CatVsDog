import torch
import os
import CatVsDog
from torchvision import transforms


mainpath = r'C:\Users\MBISFHB\Documents\code\Python Scripts\Dogsvscats'
#mainpath = '/Users/samuelbarnett/Documents/DL_data/'

trainpath = mainpath+r'\train'
testpath = mainpath+r'\test1'

testset = os.listdir(testpath)
dataset = CatVsDog.CatVsDog(testset,testpath,transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))

testloader = torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0)

net = torch.load(mainpath+'\savedmodel.pt')

correct = 0
total = 0
results = []
counter = 1
with torch.no_grad():
    for data in testloader:
        inputs = data['image']
        labels = data['annotation']
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        listed = predicted.tolist()
        results.append(listed[:])
        print(counter)
        counter+=1


print(results)