import torch
import numpy as np
from models.models_uncond import build_generator_32
from lib.rng import np_rng
from utils.inception_score import InceptionScore

inception_score = InceptionScore(2)

g = build_generator_32()
g.load_state_dict(torch.load('/content/drive/MyDrive/runs/cifar10_moegan_8_/models/gen_157.pth'))
g.cuda()

def noise_batch(self,samples=128):
    return torch.FloatTensor(np_rng.uniform(-1., 1., size=(samples, 100)))
noise = noise_batch(128).cuda()
xfake = g(noise)
IS = inception_score(xfake)
print('IS')
print('eval finished')