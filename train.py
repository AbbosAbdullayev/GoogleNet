from nn import WeatherDataset,transform,GoogleNet
import torch.nn as nnn
import torch.optim as optim
import time
import torch
import os
import torch
import torchvision
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
dataset='dataset'
data_set=WeatherDataset(dataset,transform=transform)
data_set=data_set
lengths=[int(data_set.__len__()*0.9),int(data_set.__len__()*0.1)],
#print(data_set.__getitem__(4))

trainset,validset=torch.utils.data.random_split(data_set,[1000,125])
train_loader=torch.utils.data.DataLoader(trainset,batch_size=16,shuffle=True)
valid_loader=torch.utils.data.DataLoader(validset,batch_size=16,shuffle=True)   
device= device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=GoogleNet(n_class=4).to(device)
criterion=nnn.CrossEntropyLoss()
optimzer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
train_loss,val_loss,train_cor=0,0,0
total,total_val=0,0
train_acc=[]
train_losses=[]
epochs=100
for epoch in range(epochs):
   train_loss=0.0
   start=time.time()
   for i,data in enumerate(train_loader,0):
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)
        optimzer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimzer.step()
        train_loss+=loss.item()
        _,y_pred=torch.max(outputs,1)    
        y_pred[y_pred>=0.5]=1
        y_pred[y_pred<=0.5]=0
        train_cor+=(y_pred==labels).sum().item()
        total+=labels.size(0)
        train_losses=train_loss/len(train_loader)
        train_acc=100*train_cor/total
   print(f'{time.time()-start:.3f}sec : [Epoch {epoch+1}/{epochs}] train loss: {train_loss:.3f} -> training accuracy: {train_acc:.4f}')    