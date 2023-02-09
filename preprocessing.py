import os
import torch
import torchvision
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
from nn import *
# the location of image data(main dir)
#dataset='dataset'  
# Custom dataset class
class WeatherDataset(torch.utils.data.Dataset):
    def __init__(self,dataset,transform=None):
        super(WeatherDataset,self).__init__()
        self.dataset=dataset
        self.transform=transform

        self.images=[]
        self.labels=[]

        for class_name in os.listdir(dataset):
            class_dir=os.path.join(dataset,class_name)
            if os.path.isdir(class_dir):
                if class_name=='cloudy':
                 label=0
            if class_name=='sunrise':
                label=1
            if class_name=='rainy':
                 label=2
            if class_name=='shine':
                label=3  
            for image in os.listdir(class_dir):
                image_path=os.path.join(class_dir,image)
                self.images.append(image_path)
                self.labels.append(label)    
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        image=torchvision.datasets.folder.pil_loader(self.images[idx])
        if self.transform:
           image=self.transform(image)
        return image, self.labels[idx] 
transform=transforms.Compose([
   transforms.RandomRotation(10),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

'''
data_set=WeatherDataset(dataset,transform=transform)
print(data_set.__len__())
print(data_set.__getitem__(4))
samples,labels=iter(data_set).next()
classes={0:'cloudy',1:'sunrise',2:'rainy',3:'shine'}
fig=plt.figure(figsize=(16,24))
for i in range(24):
    a=fig.add_subplot(4,6,i+1)
    a.set_title(classes[labels[i].item()])
    a.axis('off')
    a.imshow(np.transpose(samples[i].numpy(),(1,2,0)))
plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)    
'''

 
                    