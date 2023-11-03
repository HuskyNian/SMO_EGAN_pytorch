
#default paths
import os
import sys
import numpy as np 

#set modules path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from sklearn.preprocessing import OrdinalEncoder
#theano utils
from lib.rng import np_rng
from lib.data_utils import shuffle, iter_data, ImgRescale, Batch, processing_img

#plot stuff
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import time
from tqdm import tqdm
#dataset and models
from dataset.datasets import toy_dataset
from dataset.datasets import cifar10, faces, bedrooms
from models import models_uncond,models_syn
from inception_score import InceptionScore 
from frechet_inception_distance import FrechetInceptionDistance 
from metric import *
import torch
#metrics
from mmd2u import compute_metric_mmd2
import random
def set_seed(seed=42):
    os.environ["PYTHONASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class TaskDBGeneric:

    def __init__(self, name,
                       nPass,
                       nPassD,
                       nGenerators,
                       batchSize,
                       metric = "default"):
        self.name = name
        self.nPass = nPass
        print(f'training for {nPass} epochs')
        self.nPassD = nPassD
        self.batchSize = batchSize
        self.nGenerators = nGenerators
        self.noiseSize = 100
        self.miniBatchForD = int(batchSize/nGenerators*nPassD)
        print('batch size for each generator',self.miniBatchForD)
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
        nit = int(self.X_train.shape[0] / size)   # -1 means drop last batch
        for p in range(self.nPass): 
            self.X_train = shuffle(self.X_train) # shuffle the training dataset at each epoch
            for it in range(nit):
                self.lastBatch = self.X_train[it*size:(it+1)*size,]
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
                img = gen_dis[r*grid+c,]
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
        xfake_list = instances[0].img[0:self.miniBatchForD]
        for i in range(1,len(instances)):
            xfake = instances[i].img[0:self.miniBatchForD]
            xfake_list = np.append(xfake_list, xfake, axis=0)
        #iteration
        for xreal, xfake in iter_data(xreals, shuffle(xfake_list), size=self.batchSize):
            yield xreal, xfake

    def statistic_datas(self, fakeset):
        return  self.X_test[0:self.batchSize], fakeset[0:self.batchSize]
    
    def compute_metrics(self, instances, generator, in_samples=128):
        #metric set
        samples = min(in_samples,  self.X_test.shape[0])
        s_zmb = self.noise_batch(min(128,samples))
        print('n samples',s_zmb.shape)
        #create test batch
        if self.frechet_inception_distance is not None:
            test_metric_batch = self.X_test[0:samples]
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

class TaskSynthetic(TaskDBGeneric):

    def __init__(self, name,
                       nPass,
                       nPassD,
                       nGenerators,
                       batchSize,
                       dataset,
                       data_sampler,
                       data_transformer,
                       embedding_dim,
                       metric = "default",df=None):
        super().__init__(name,
                         nPass,
                         nPassD,
                         nGenerators,
                         batchSize,
                         metric)
        self.generator_builder = models_syn.build_generator_syn
        self.discriminator_builder = models_syn.build_discriminator_syn
        self.sampler = data_sampler
        self.transformer = data_transformer
        self.imagesize = 32 # saved image size
        self.embedding_dim = embedding_dim
        #self.dataset = dataset
        self.X_train = dataset
        #self.X_test = self.dataset["X_test"]
        self.device='cuda'
        self.mean = torch.zeros(batchSize, self.embedding_dim, device=self.device)
        self.std = self.mean +1
        self.out_dim = None
        self.orig_df = df
        if name=='UK':
            self.ordinal_encoder = self.make_ordinal_encoder(df)
        else:
            self.ordinal_encoder = None

    def make_ordinal_encoder(self,orig):
        key_cols = ['AGE','ECONPRIM','ETHGROUP','LTILL','QUALNUM','SEX','SOCLASS','TENURE','MSTATUS']
        # get columns used and make y to be binary
        orig = orig[key_cols]
        orig['MSTATUS'] = (orig['MSTATUS'] == 'Married' ) | (orig['MSTATUS'] == 'Remarried' )
        orig['TENURE'] = (orig['TENURE'] == 'Own occ-buying' ) | (orig['TENURE'] == 'Own occ-outright' )
        encoder = OrdinalEncoder()
        encoder.fit(orig)
        return encoder

    def is_last (self):
        size = self.batchSize*self.nPassD
        nit = int(self.X_train.shape[0] / size)-1
        return self.index == (self.nPass-1)*nit+(nit-1)

    def get_range(self):
        size = self.batchSize*self.nPassD
        nit = int(self.X_train.shape[0] / size) -1 # drop last batch
        print(self.name + ' total updates:',self.nPass,nit,self.nPass*nit)
        for p in range(self.nPass): 
            for it in range(nit):
                self.lastBatch = self.X_train[it*size:(it+1)*size]
                self.index = p*nit+it
                yield self.index
    
    def create_geneator(self,generator_dim = (256, 256)):
        generator = self.generator_builder(self.sampler,self.transformer,self.embedding_dim,generator_dim)
        return generator

    def create_discriminator(self, discriminator_dim = (256, 256),pac = 10):
        discriminator = self.discriminator_builder(self.sampler,self.transformer, discriminator_dim,pac)
        return discriminator
    
    def get_fakez(self,n): # n : BatchSize
        mean = torch.zeros(n, self.embedding_dim,device=self.device)
        fakez = torch.normal(mean=mean, std=mean+1)
        return fakez

    def evaluate(self, instances, generator):
        
        #compue is for all points
        tcap_all = []
        cio_all = []
        uni_roc_all = []
        bi_roc_all = []
        with torch.no_grad():
            
            for i in range(0, len(instances)):
                print('start next instance \n')
                for k in range(5):
                    start = time.time()
                    set_seed(42+k)
                    generator.set(torch.load(instances[i]))
                    #generator.generator.eval()
                    xfake = (generator.gen( self.noise_batch(len(self.orig_df)).cuda())  )
                    xfake = xfake.cpu().detach().numpy()
                    assert len(xfake)==len(self.orig_df)
                    xfake = self.transformer.inverse_transform(xfake)
                    csv_path = os.path.dirname(instances[i])
                    xfake.to_csv(os.path.join(csv_path,f'generation_{i}.csv'))
                    tcap = cal_mean_tcap(self.name,self.orig_df,xfake)
                    tcap_all.append(tcap)
                    cio = cal_mean_cio(self.name,self.orig_df,xfake,self.ordinal_encoder)
                    cio_all.append(cio)
                    uni_roc,bi_roc = cal_mean_roc(self.name,self.orig_df,xfake)
                    uni_roc_all.append(uni_roc)
                    bi_roc_all.append(bi_roc)
                    print('evaluate time',time.time()-start)
                    print('tcap',tcap,'utility',(bi_roc+uni_roc+cio)/3,'cio,uni_roc,bi_roc',
                    cio,uni_roc,bi_roc)
        return {'tcap':np.array(tcap_all),
                'cio':np.array(cio_all),
                'uni_roc':np.array(uni_roc_all),
                'bi_roc':np.array(bi_roc_all),}

    def compute_metrics(self, instances, generator, in_samples=128):
        #metric set
        #print('by default, generate the shape 1024')
        xfake_list = []
        #steps = int(np.ceil(len(self.X_train)/1024))
        steps =1
        #compue is for all points
        tcap_all = []
        cio_all = []
        uni_roc_all = []
        bi_roc_all = []
        start = time.time()
        with torch.no_grad():
            for i in range(0, len(instances)):
                #for step in range(steps):  ###   5600      6
                #    if step==steps-1:
                #        xfake_list.append(generator(instances[i], self.noise_batch(len(self.X_train)-step*1024 ).cuda())  )
                #    else:
                generator.set(instances[i].params)
                generator.generator.eval()
                #xfake_list.append(generator.gen( self.noise_batch(500).cuda())  )
                #xfake = generator.gen( self.noise_batch(6600).cuda())
                xfake = generator.gen( self.noise_batch(len(self.orig_df)).cuda())
                generator.generator.train()
                xfake = xfake.cpu().detach().numpy()
                xfake = self.transformer.inverse_transform(xfake)
                tcap_all.append(cal_mean_tcap(self.name,self.orig_df,xfake))
                cio_all.append(cal_mean_cio(self.name,self.orig_df,xfake,self.ordinal_encoder))
                uni_roc,bi_roc = cal_mean_roc(self.name,self.orig_df,xfake)
                uni_roc_all.append(uni_roc)
                bi_roc_all.append(bi_roc)
        print('evaluate time',time.time()-start)
        return {'tcap':np.array(tcap_all),
                'cio':np.array(cio_all),
                'uni_roc':np.array(uni_roc_all),
                'bi_roc':np.array(bi_roc_all),}

    def noise_batch(self, n=None, condition_column=None, condition_value=None):
        """Sample data similar to the training data.
        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        Returns:
            torch.tensor
                A tensor of noise
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self.transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self.sampler.generate_cond_from_condition_column_info(
                condition_info, self.batchSize)
        else:
            global_condition_vec = None

        
        data = []
        if n is None:
            n= self.batchSize
        mean = torch.zeros(n, self.embedding_dim)
        std = mean + 1
        fakez = torch.normal(mean=mean, std=std).to(self.device)

        if global_condition_vec is not None:
            condvec = global_condition_vec.copy()
        else:
            condvec = self.sampler.sample_original_condvec(n)

        if condvec is None:
            pass
        else:
            c1 = condvec
            c1 = torch.from_numpy(c1).to(self.device)
            fakez = torch.cat([fakez, c1], dim=1)
        return fakez
    
    def iter_data_discriminator(self, instances):
        #prepare
        if self.batchSize == self.miniBatchForD: 
            xreal_list = instances[0].xreal
            xfake_list = instances[0].xfake
        else:
            indexes = np.random.randint(0,self.batchSize,self.miniBatchForD)
            xreal_list = instances[0].xreal[indexes]
            xfake_list = instances[0].xfake[indexes]
            for i in range(1,len(instances)):
                if i == len(instances)-1:
                    num = self.batchSize - len(xfake_list)
                    indexes = np.random.randint(0,self.batchSize,num)
                    xfake = instances[i].xfake[indexes]
                    xreal = instances[i].xreal[indexes]
                else:
                    xfake = instances[i].xfake[indexes]
                    xreal = instances[i].xreal[indexes]
                xfake_list = torch.cat([xfake_list, xfake], axis=0)
                xreal_list = torch.cat([xreal_list, xreal], axis=0)
        for xreal, xfake in iter_data(xreal_list, xfake_list, size=self.batchSize):
            yield xreal, xfake

