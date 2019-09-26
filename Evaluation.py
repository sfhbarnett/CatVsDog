import torch
import os
import CatVsDog
from torchvision import transforms
import csv


mainpath = r'C:\Users\MBISFHB\Documents\code\Python Scripts\Dogsvscats'
#mainpath = '/Users/samuelbarnett/Documents/DL_data/'

trainpath = mainpath+r'\train'
testpath = mainpath+r'test'

testset = os.listdir(testpath)
dataset = CatVsDog.CatVsDog(testset,testpath,transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]),train=0)

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
        for i in range(4):
            results.append([labels[i][:-4],listed[i]])
        print(counter)
        counter+=1
c = 1
print(results)
with open('results.csv', mode='w', newline='') as resultsfile:
    resultswriter = csv.writer(resultsfile, delimiter=',')
    resultswriter.writerow(['id', 'label'])
    for x in range(len(results)):
        resultswriter.writerow(results[x])
        c +=1
print("done")