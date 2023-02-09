import torch
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot

## device
#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Inception(nn.Module):
    def __init__(self,input_ch,out1x1,red3x3,out3x3,red5x5,out5x5,pool_proj):
        super(Inception,self).__init__()
        self.arm1x1=nn.Sequential(
            nn.Conv2d(input_ch,out1x1,kernel_size=1),
            nn.BatchNorm2d(out1x1),
            nn.ReLU(True),
        )

        self.arm3x3=nn.Sequential(
            nn.Conv2d(input_ch,red3x3,kernel_size=1),
            nn.BatchNorm2d(red3x3),
            nn.ReLU(True),
            nn.Conv2d(red3x3,out3x3,kernel_size=3,padding=1),
            nn.BatchNorm2d(out3x3),
            nn.ReLU(True),
        )
       
        self.arm5x5=nn.Sequential(
            nn.Conv2d(input_ch,red5x5,kernel_size=1),
            nn.BatchNorm2d(red5x5),
            nn.ReLU(True),
            nn.Conv2d(red5x5,out5x5,5,padding=2),
            nn.BatchNorm2d(out5x5),
            nn.ReLU(True),
        )       
 
        self.arm_pool=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(input_ch,pool_proj,kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(True)
        )
    def forward(self,m):
        arm1=self.arm1x1(m)
        arm3=self.arm3x3(m)
        arm5=self.arm5x5(m)
        pool_proj=self.arm_pool(m)
        return torch.cat([arm1,arm3,arm5,pool_proj],1)

class GoogleNet(nn.Module):
    def __init__(self,n_class):
        super(GoogleNet,self).__init__()
        self.conv_block=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.inception3a=Inception(192,64,96,128,16,32,32)
        self.inception3b=Inception(256,128,128,192,32,96,64)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.inception4a=Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b=Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c=Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d=Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e=Inception(528, 256, 160, 320, 32, 128, 128)
        self.inception5a=Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b=Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool=nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout=nn.Dropout(0.4)
        self.fc=nn.Linear(1024,n_class)

    def forward(self,m):
        m=self.conv_block(m)
        m=self.inception3a(m)
        m=self.inception3b(m)
        m=self.maxpool(m)
        m=self.inception4a(m)
        m=self.inception4b(m)
        m=self.inception4c(m)
        m=self.inception4d(m)
        m=self.inception4e(m)
        m=self.maxpool(m)
        m=self.inception5a(m)
        m=self.inception5b(m)
        m=self.avgpool(m)
        m=m.view(m.size(0),-1)
        m=self.dropout(m)
        m=self.fc(m)
        return m
if __name__=='__main__':
    #batch=4
#x=torch.randn(batch,3,224,224)
    model=GoogleNet(n_class=2)
    summary(model,input_size=(3,224,224))
    dumpy_input=torch.randn(1,3,224,224)
    graph=make_dot(model(dumpy_input),params=dict(model.named_parameters()))
  #  graph.render('GoogleNet')
#model.summary()
#print(model(x)[2].shape)
