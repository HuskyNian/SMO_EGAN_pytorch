
#default paths
import os
import sys
import numpy as np 

#set modules path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

#theano utils
from lib.rng import np_rng
from lib.data_utils import shuffle, iter_data, ImgRescale, Batch, processing_img

#plot stuff
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

#dataset and models
from dataset.datasets import toy_dataset
from dataset.datasets import cifar10, faces, bedrooms
from models import models_uncond
from inception_score import InceptionScore 
from frechet_inception_distance import FrechetInceptionDistance 
import torch
#metrics
from mmd2u import compute_metric_mmd2

class Task:
    def __init__(self,name,
                      nPass,
                      nPassD,
                      nGenerators,
                      batchSize,
                      metric = 'default'):
        self.name = name
    
    def create_geneator(self, noise, dim):
        pass 
    
    def create_discriminator(self, dim, GP_norm):
        pass 
    
    def is_last (self):
        pass 

    def get_range(self):
        pass 

    def batch(self):
        pass

    def noise_batch(self, samples=None):
        pass

    def save_image(self, true_dis, gen_dis, path_images, name):
        pass

    def iter_data_discriminator(self, xreals, instances):
        pass

    def statistic_datas(self, instance):
        pass

    def compute_metrics(self, instance):
        pass

    def net_output_type(self):
        #return lambda x : T.matrix(x)
        raise "not implemented net_output_type in task.py"

    def __str__(self):
        return "None"

class TaskTrainToy:

    def __init__(self, name,
                       nPass,
                       nPassD,
                       nGenerators,
                       batchSize,
                       metric = 'default'):
        self.name = name
        self.nPass = nPass
        self.nPassD = nPassD # 1
        self.batchSize = batchSize
        self.nGenerators = nGenerators
        self.noiseSize = 2
        self.miniBatchForD = int(batchSize/nGenerators*nPassD)
        self.lastBatch = None
        self.index = 0

    def create_geneator(self, noise, dim):
        generator = models_uncond.build_generator_toy(noise, nd=dim)
        return generator

    def create_discriminator(self, dim, GP_norm):
        discriminator = models_uncond.build_discriminator_toy(nd=dim, GP_norm=GP_norm)
        return discriminator


    def is_last (self):
        return self.index == self.nPass-1

    def get_range(self):
        print('n Pass:',self.nPass)
        for i in range(self.nPass): 
            self.index = i
            yield i

    def batch(self):
        size = self.batchSize*self.nPassD
        xmb = toy_dataset(DATASET=self.name, size=size)
        if xmb.shape[0] == size:
            self.lastBatch = xmb
        else:
            self.lastBatch = xmb[:size]
        return self.lastBatch

    def save_image(self, true_dis, gen_dis, path_images, name):
        plt.clf()
        plt.scatter(true_dis[:, 0], true_dis[:, 1], c='orange', marker='+')
        plt.scatter(gen_dis[:, 0], gen_dis[:, 1], c='green', marker='+')
        plt.savefig(os.path.join(path_images, name + '.jpg'))

    def noise_batch(self,samples=None):
        if samples==None:
            #return floatX(np_rng.uniform(-1., 1., size=(self.batchSize, self.noiseSize)))
            return torch.FloatTensor(np_rng.uniform(-1., 1., size=(self.batchSize, self.noiseSize)))
        #return floatX(np_rng.uniform(-1., 1., size=(samples, self.noiseSize)))
        return torch.FloatTensor(np_rng.uniform(-1., 1., size=(samples, self.noiseSize)))

    def iter_data_discriminator(self, xreals, instances):
        #prepare
        xfake_list = instances[0].img[0:self.miniBatchForD, :]
        for i in range(1,len(instances)):
            xfake = instances[i].img[0:self.miniBatchForD, :]
            #xfake_list = np.append(xfake_list, xfake, axis=0)
            xfake_list = torch.cat([xfake_list, xfake], dim=0)
        #iteration
        for xreal, xfake in iter_data(xreals, shuffle(xfake_list), size=self.batchSize):
            yield xreal, xfake

    def statistic_datas(self, fakeset):
        sample_xmb = toy_dataset(DATASET=self.name, size=self.batchSize)
        return  sample_xmb[0:self.batchSize, :], fakeset[0:self.batchSize, :]
    
    def compute_metrics(self, instances, generator, samples=512):
        #metric set
        s_zmb = self.noise_batch(samples)
        xreal = toy_dataset(DATASET=self.name, size=samples)
        #compue mmd for all points
        mmd2_all = []
        for i in range(0, len(instances)):
            xfake = generator(instances[i], s_zmb)
            mmd2_all.append(abs(compute_metric_mmd2(xfake,xreal)))
        #return
        return np.array(mmd2_all)

    def net_output_type(self):
        #return lambda x : T.matrix(x)
        raise "not implemented net_output_type in task.py"

    def __str__(self):
        return self.name

class TaskDBGeneric:

    def __init__(self, name,
                       nPass,
                       nPassD,
                       nGenerators,
                       batchSize,
                       metric = "default"):
        self.name = name
        self.nPass = nPass
        self.nPassD = nPassD
        self.batchSize = batchSize
        self.nGenerators = nGenerators
        self.noiseSize = 100
        self.miniBatchForD = int(batchSize/nGenerators*nPassD)
        self.lastBatch = None
        self.index = 0
        self.inception_score = None
        self.frechet_inception_distance = None
        if metric.lower() in ("default", "is"):
            self.inception_score = InceptionScore(2)
        elif metric.lower() in ("fid"):
            self.frechet_inception_distance = FrechetInceptionDistance()
        else: 
            raise "Invalid metric"

    def create_geneator(self, noise, dim):
        generator = self.generator_builder(noise, ngf=dim)
        return generator

    def create_discriminator(self, dim, GP_norm):
        discriminator = self.discriminator_builder(ndf=dim)
        return discriminator

    def is_last (self):
        size = self.batchSize*self.nPassD
        nit = int(self.X_train.shape[0] / size)
        return self.index == (self.nPass-1)*nit+(nit-1)

    def get_range(self):
        size = self.batchSize*self.nPassD
        nit = int(self.X_train.shape[0] / size)
        for p in range(self.nPass): 
            for it in range(nit):
                self.lastBatch = self.X_train[it*size:(it+1)*size, :, :, :]
                #self.lastBatch = Batch(self.lastBatch, self.imagesize, 0, flip=True) 
                self.lastBatch = processing_img(self.lastBatch, center=True, scale=True, convert=False)
                self.index = p*nit+it
                yield self.index

    def batch(self):
        return self.lastBatch

    def save_image(self, true_dis, gen_dis, path_images, name):
        grid = 8
        blank_image = Image.new("RGB",(self.imagesize*grid+9,self.imagesize*grid+9))
        for r in range(grid):
            for c in range(grid):
                img = gen_dis[r*grid+c,:,:,:]
                img = ImgRescale(img, center=True, scale=True, convert_back=True)
                blank_image.paste(Image.fromarray(img),(c*self.imagesize+c+1,r*self.imagesize+r+1)) 
        blank_image.save(os.path.join(path_images, name + '.png'))

    def noise_batch(self,samples=None):
        if samples==None:
            #return floatX(np_rng.uniform(-1., 1., size=(self.batchSize, self.noiseSize)))
            return torch.FloatTensor(np_rng.uniform(-1., 1., size=(self.batchSize, self.noiseSize)))
        #return floatX(np_rng.uniform(-1., 1., size=(samples, self.noiseSize)))
        return torch.FloatTensor(np_rng.uniform(-1., 1., size=(samples, self.noiseSize)))

    def iter_data_discriminator(self, xreals, instances):
        #prepare
        xfake_list = instances[0].img[0:self.miniBatchForD, :, :, :]
        for i in range(1,len(instances)):
            xfake = instances[i].img[0:self.miniBatchForD, :, :, :]
            xfake_list = np.append(xfake_list, xfake, axis=0)
        #iteration
        for xreal, xfake in iter_data(xreals, shuffle(xfake_list), size=self.batchSize):
            yield xreal, xfake

    def statistic_datas(self, fakeset):
        return  self.X_test[0:self.batchSize, :, :, :], fakeset[0:self.batchSize, :, :, :]
    
    def compute_metrics(self, instances, generator, in_samples=128):
        #metric set
        samples = min(in_samples,  self.X_test.shape[0])
        s_zmb = self.noise_batch(min(128,samples))
        print('n samples',s_zmb.shape)
        #create test batch
        if self.frechet_inception_distance is not None:
            test_metric_batch = self.X_test[0:samples, :, :, :]
        #compue is for all points
        is_all = []
        for i in range(0, len(instances)):
            xfake = generator(instances[i], s_zmb)
            if self.inception_score is not None:
                is_all.append(-self.inception_score(xfake)[0])
            elif self.frechet_inception_distance is not None:
                is_all.append(self.frechet_inception_distance(xfake, test_metric_batch))
        #return
        return np.array(is_all)

    def net_output_type(self):
        #return lambda x : T.tensor4(x)
        raise "not implemented net_output_type in task.py"

    def __str__(self):
        return self.name

class TaskCifar10(TaskDBGeneric):

    def __init__(self, name,
                       nPass,
                       nPassD,
                       nGenerators,
                       batchSize,
                       metric = "default"):
        super().__init__(name,
                         nPass,
                         nPassD,
                         nGenerators,
                         batchSize,
                         metric)
        self.generator_builder = models_uncond.build_generator_32
        self.discriminator_builder = models_uncond.build_discriminator_32
        self.imagesize = 32
        self.dataset = cifar10()
        self.X_train = shuffle(self.dataset["X_train"])
        self.X_test = self.dataset["X_test"]

    def get_range(self):
        size = self.batchSize*self.nPassD
        nit = int(self.X_train.shape[0] / size)
        print('cifar 10 total updates:',self.nPass,nit,self.nPass*nit)
        for p in range(self.nPass): 
            for it in range(nit):
                self.lastBatch = self.X_train[it*size:(it+1)*size, :, :, :]
                self.index = p*nit+it
                yield self.index

class TaskFaces(TaskDBGeneric):

    def __init__(self, name,
                       nPass,
                       nPassD,
                       nGenerators,
                       batchSize,
                       metric = "default"):
        super().__init__(name,
                         nPass,
                         nPassD,
                         nGenerators,
                         batchSize,
                         metric)
        self.generator_builder = models_uncond.build_generator_64
        self.discriminator_builder = models_uncond.build_discriminator_64
        self.imagesize = 64
        self.dataset = faces(self.imagesize)
        self.X_train = shuffle(self.dataset)
        self.X_test = shuffle(self.dataset)[0:1024]

class TaskBedrooms(TaskDBGeneric):

    def __init__(self, name,
                       nPass,
                       nPassD,
                       nGenerators,
                       batchSize,
                       metric = "default"):
        super().__init__(name,
                         nPass,
                         nPassD,
                         nGenerators,
                         batchSize,
                         metric)
        self.generator_builder = models_uncond.build_generator_64 #128
        self.discriminator_builder = models_uncond.build_discriminator_64 #128
        self.imagesize = 64 #128
        self.dataset = bedrooms(self.imagesize)
        self.X_train = shuffle(self.dataset)
        self.X_test = shuffle(self.dataset)[0:1024]
