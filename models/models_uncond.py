import torch
import torch.nn as nn 
def DenseLayer(in_ch,out,act=nn.ReLU()):
    layers = nn.Sequential(
        nn.Linear(in_ch,out),
        act
    )
    return layers

def Deconv2DLayer(in_ch,out,kernel=4,stride=2,padding=1,act=nn.ReLU()):
    layers = nn.Sequential(
        nn.ConvTranspose2d(in_ch,out,kernel,stride,padding=padding),
        act
    )
    return layers

def init_weights(module):
    if isinstance(module,(nn.Conv2d,nn.Linear,nn.ConvTranspose2d)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
            
def build_generator_toy(noise=None,nd=512):
    return Generator_toy(nd)
def build_discriminator_toy(image=None, nd=512, GP_norm=None):
    return Discriminator_toy(nd,GP_norm)
def build_generator_32(noise=None, ngf=128):
    return Generator_32(ngf)
def build_discriminator_32(image=None,ndf=128):
    return Discriminator_32(ndf)
def build_generator_64(noise=None, ngf=128):
    return Generator_64(ngf)
def build_discriminator_64(image=None,ndf=128):
    return Discriminator_64(ndf)
def build_generator_128(noise=None, ngf=128):
    return Generator_128(ngf)
def build_discriminator_128(image=None,ndf=128):
    return Discriminator_128(ndf)
class Generator_toy(nn.Module):
    ''' 
        input shape (bs,100) => (bs,3,64,64)
    '''
    def __init__(self,nd=512):
        super().__init__()
        self.nd = nd
        act = nn.ReLU()
        self.layers = nn.Sequential(nn.Linear(2,nd),act,
                                    nn.Linear(nd,nd),act,
                                    nn.Linear(nd,nd),act,
                                    nn.Linear(nd,2),
                                    )
        self.apply(init_weights)

    def forward(self,x):
        out = self.layers(x)
        return out 

class Discriminator_toy(nn.Module):
    '''
       input shape (bs,3,32,32) = (bs,512,4,4)
    '''
    def __init__(self,nd=512,GP_norm=None):
        super().__init__()
        self.nd = nd
        act = nn.ReLU()
        self.layers = nn.Sequential(nn.Linear(2,nd),act)
        for _ in range(2):
            if GP_norm is True:
                self.layers.append(nn.Linear(nd,nd))
                self.layers.append(act)
            else:
                self.layers.append(nn.Linear(nd,nd))
                self.layers.append(act)
                self.layers.append(nn.BatchNorm1d(nd))
            
        self.fc = nn.Sequential(nn.Linear(nd,1),nn.Sigmoid())
        self.apply(init_weights)

    def forward(self,x):
        out = self.layers(x)
        out = self.fc(out)
        return out    

class Generator_32(nn.Module):
    '''
        (bs,100)=> (bs,3,32,32)
    '''
    def __init__(self,ngf=128):
        super().__init__()
        self.ngf = ngf
        self.dense1 = DenseLayer(100,ngf*4*4*4)
        self.deconv1 = Deconv2DLayer(ngf*4,ngf*2)
        self.deconv2 = Deconv2DLayer(ngf*2,ngf)
        self.deconv3 = Deconv2DLayer(ngf,3,act=nn.Tanh())
        self.apply(init_weights)

    def forward(self,x):
        b,nd = x.shape
        out = self.dense1(x)
        out = out.reshape([b,self.ngf*4,4,4])
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        return out

class Discriminator_32(nn.Module):
    '''
       input shape (bs,3,32,32) = (bs,512,4,4)
    '''
    def __init__(self,ndf=128):
        super().__init__()
        self.ndf = ndf
        act = nn.LeakyReLU(0.2)
        self.layers = nn.Sequential(nn.Conv2d(3,ndf,4,stride=2,padding=1),act,
                                    nn.Conv2d(ndf,ndf*2,4,stride=2,padding=1),act,nn.BatchNorm2d(ndf*2),
                                    nn.Conv2d(ndf*2,ndf*4,4,stride=2,padding=1),act,nn.BatchNorm2d(ndf*4),
                                    )
        self.fc = nn.Sequential(nn.Linear(ndf*4*4*4,1),nn.Sigmoid())
        self.apply(init_weights)

    def forward(self,x):
        out = self.layers(x)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out

class Generator_64(nn.Module):
    ''' 
        input shape (bs,100) => (bs,3,64,64)
    '''
    def __init__(self,ngf=128):
        super().__init__()
        self.ngf = ngf
        act = nn.LeakyReLU(0.2)
        self.dense1 = DenseLayer(100,ngf*8*4*4)
        self.layers = nn.Sequential(Deconv2DLayer(ngf*8,ngf*8),
                                    Deconv2DLayer(ngf*8,ngf*4),
                                    Deconv2DLayer(ngf*4,ngf*4),
                                    Deconv2DLayer(ngf*4,ngf*2),
                                    Deconv2DLayer(ngf*2,3,kernel=3,stride=1,act=nn.Tanh()),
                                    )
        self.apply(init_weights)

    def forward(self,x):
        b,nd = x.shape
        out = self.dense1(x)
        out = out.reshape([b,self.ngf*8,4,4])
        out = self.layers(out)
        return out     

class Discriminator_64(nn.Module):
    '''
        input shape(bs,3,64,64) => (bs,1024,4,4) => (bs,1)
    '''
    def __init__(self,ndf=128):
        super().__init__()
        self.ndf = ndf
        act = nn.LeakyReLU(0.2)
        self.layers = nn.Sequential(nn.Conv2d(3,ndf,4,stride=2,padding=1),act,
                                    nn.Conv2d(ndf,ndf*2,4,stride=2,padding=1),act,nn.BatchNorm2d(ndf*2),
                                    nn.Conv2d(ndf*2,ndf*4,4,stride=2,padding=1),act,nn.BatchNorm2d(ndf*4),
                                    nn.Conv2d(ndf*4,ndf*8,4,stride=2,padding=1),act,nn.BatchNorm2d(ndf*8),
                                    )
        self.fc = nn.Sequential(nn.Linear(ndf*8*4*4,1),nn.Sigmoid())
        self.apply(init_weights)

    def forward(self,x):
        out = self.layers(x)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out

class Generator_128(nn.Module):
    ''' 
        input shape (bs,100) => (bs,3,128,128)
    '''
    def __init__(self,ngf=128):
        super().__init__()
        self.ngf = ngf
        act = nn.LeakyReLU(0.2)
        self.dense1 = DenseLayer(100,ngf*16*4*4)
        self.layers = nn.Sequential(Deconv2DLayer(ngf*16,ngf*8),
                                    Deconv2DLayer(ngf*8,ngf*8),
                                    Deconv2DLayer(ngf*8,ngf*4),
                                    Deconv2DLayer(ngf*4,ngf*4),
                                    Deconv2DLayer(ngf*4,ngf*2),
                                    Deconv2DLayer(ngf*2,3,kernel=3,stride=1,act=nn.Tanh()),
                                    )
        self.apply(init_weights)

    def forward(self,x):
        b,nd = x.shape
        out = self.dense1(x)
        out = out.reshape([b,self.ngf*16,4,4])
        out = self.layers(out)
        return out 

class Discriminator_128(nn.Module):
    '''
        input shape(bs,3,128,128) => (bs,2048,4,4) => (bs,1)
    '''
    def __init__(self,ndf=128):
        super().__init__()
        self.ndf = ndf
        act = nn.LeakyReLU(0.2)
        self.layers = nn.Sequential(nn.Conv2d(3,ndf,4,stride=2,padding=1),act,
                                    nn.Conv2d(ndf,ndf*2,4,stride=2,padding=1),act,nn.BatchNorm2d(ndf*2),
                                    nn.Conv2d(ndf*2,ndf*4,4,stride=2,padding=1),act,nn.BatchNorm2d(ndf*4),
                                    nn.Conv2d(ndf*4,ndf*8,4,stride=2,padding=1),act,nn.BatchNorm2d(ndf*8),
                                    nn.Conv2d(ndf*8,ndf*16,4,stride=2,padding=1),act,nn.BatchNorm2d(ndf*16),
                                    )
        self.fc = nn.Sequential(nn.Linear(ndf*16*4*4,1),nn.Sigmoid())
        self.apply(init_weights)

    def forward(self,x):
        out = self.layers(x)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out    