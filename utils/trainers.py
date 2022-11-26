import torch
import torch.nn as nn
import torch.nn.functional as F
from .mab import get_output
class GeneratorTrainer:
    def __init__(self, noise, generator, discriminator, lr, b1):
        #self.noise=noise
        
        self.generator=generator
        self.discriminator=discriminator
        
        self.generator_params = self.generator.parameters()
        self.optimizer = torch.optim.Adam(self.generator_params,lr=lr,betas=(b1,0.999))
    
    def gen_fn(self,noise):
        out = get_output(self.generator,noise,train=False)
        return out
    
    def _train(self,noise):
        noise = noise.cuda()
        Tgimgs = get_output(self.generator,noise,train=True)
        Tfake_out = get_output(self.discriminator,Tgimgs,train=True)
        return Tfake_out
        
    def train_g(self,noise): 
        Tfake_out = self._train(noise)
        loss = F.binary_cross_entropy(Tfake_out, torch.ones_like(Tfake_out))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def train_g_minimax(self,noise):
        Tfake_out = self._train(noise)
        loss = -F.binary_cross_entropy(Tfake_out, torch.zeros_like(Tfake_out))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def train_g_ls(self,noise):
        Tfake_out = self._train(noise)
        loss = torch.mean(torch.square(Tfake_out-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    
    def train(self,loss_type,zmb):
        if loss_type == 'trickLogD':
            return self.train_g(zmb)
        elif loss_type == 'minimax':
            return self.train_g_minimax(zmb)
        elif loss_type == 'ls':
            return self.train_g_ls(zmb)
        else:
            raise "{} is invalid loss".format(loss_type)

    def gen(self,zmb):
        zmb = zmb.cuda()
        return self.gen_fn(zmb).cpu()

    def set(self,params):
        self.generator.load_state_dict(params)
        return self

    def get(self):
        return self.generator.state_dict()

