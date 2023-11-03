import torch
import torch.nn as nn
import torch.nn.functional as F
from .mab import get_output
from packaging import version
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
class GeneratorTrainer:
    def __init__(self, transformer, generator, discriminator, lr, b1):
        self.generator=generator
        self.discriminator=discriminator
        self.transformer = transformer
        self.optimizer = torch.optim.Adam(self.generator.parameters(),lr=lr,betas=(b1,0.9),weight_decay=1e-6)
    
    def gen_fn(self,noise):
        #out = get_output(self.generator,noise,train=False)
        out = self.generator(noise)
        return out
    
    def apply_activate(self,data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self.transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)
        
    def cond_loss(self,data,c,m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self.transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = F.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]
    
    def _train(self,noise):
        fakez,c1,m1 = noise
        fake = self.generator(fakez)
        fakeact = self.apply_activate(fake)
        if c1 is None:
            y_fake= self.discriminator(fakeact)
            cross_entropy=0
        else:
            y_fake = self.discriminator(torch.cat([fakeact,c1],dim=1))
            cross_entropy = self.cond_loss(fake,c1,m1) 
        return y_fake,cross_entropy,fakeact  ## weighted 0.08 to cond loss
        
    def train_g(self,noise): 
        Tfake_out,cond_loss,fakeact = self._train(noise)
        loss = F.binary_cross_entropy(Tfake_out, torch.ones_like(Tfake_out)) +cond_loss
        #loss = torch.mean(Tfake_out) +cond_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return fakeact,cond_loss.cpu().detach().numpy(),(loss-cond_loss).cpu().detach().numpy()
        
    def train_g_minimax(self,noise):
        Tfake_out,cond_loss,fakeact = self._train(noise)
        loss = -F.binary_cross_entropy(Tfake_out, torch.zeros_like(Tfake_out))+cond_loss
        #loss = -torch.mean(Tfake_out) + cond_loss
        self.optimizer.zero_grad()
        loss.backward()
      
        self.optimizer.step()
        return fakeact,cond_loss.cpu().detach().numpy(),(loss-cond_loss).cpu().detach().numpy()
        
    def train_g_ls(self,noise):
        Tfake_out,cond_loss,fakeact = self._train(noise)
        loss = torch.mean(torch.square(Tfake_out-1))+cond_loss # TODO, if not sigmoid, what can we do to replace -1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return fakeact,cond_loss.cpu().detach().numpy(),(loss-cond_loss).cpu().detach().numpy()
    
    def train(self,loss_type,zmb):
        if loss_type == 'trickLogD':
            return self.train_g(zmb)
        elif loss_type == 'minimax':
            return self.train_g_minimax(zmb)
        elif loss_type == 'ls':
            return self.train_g_ls(zmb)
        else:
            raise "{} is invalid loss".format(loss_type)

    def gen(self,fakez,c1=None):
        '''
        Parameters
        ----------
        fakez : torch.tensor
            noise.
        c1 : torch.tenspr
            conditional vector. if fakez is consisted of c1, then c1 here should give None

        Returns
        -------
        fake : TYPE
            fake tabular data after the activation

        '''
        
        if c1 is not None:
            fakez = torch.cat([fakez,c1],dim=1)
        fake = self.gen_fn(fakez)
        fake = self.apply_activate(fake)
        return fake

    def set(self,params):
        self.generator.load_state_dict(params)

    def get(self):
        return copy.deepcopy(self.generator.state_dict())
    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.
        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse('1.2.0'):
            for i in range(10):
                transformed = F.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')

        return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
        
