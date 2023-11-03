#standard stuff
import os
import argparse
import random
import json
from re import X
import numpy as np
import pandas as pd
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
#theano utils
from lib.data_utils import shuffle, iter_data
from sklearn.model_selection import train_test_split
#train utils
from utils.mab import get_output
from utils.trainers import GeneratorTrainer
from utils.nsga2 import nsga_2_pass
from utils.mmd2u import compute_metric_mmd2
from utils.variation import get_varation, get_varation_names
from utils.task import TaskSynthetic
from utils.timer import Timer
from utils.log import Logger
from utils.metric import *
from utils.auto_normalization import AutoNormalization
from models.data_sampler import DataSampler
from models.data_transformer import DataTransformer

#plot stuff
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def print_g(gs,old=False):
    if len(gs) == 0:
        print('old' if old else '', 'generator indexes: empty')
        return
    indexes = [i.index for i in gs]
    print('old' if old else '','generator indexes:',indexes)

def set_seed(seed=42):
    os.environ["PYTHONASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def build_output_dirs(output_dir,description_name):
    paths = (
        os.path.join(output_dir),
        os.path.join(output_dir,description_name),
        os.path.join(output_dir,description_name,'front'),
        os.path.join(output_dir,description_name,'logs'),
        os.path.join(output_dir,description_name,'models'),
        os.path.join(output_dir,description_name,'models','last'),
        os.path.join(output_dir,description_name,'imgs'),
    )
    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)

    return (*paths[2:], )

class Instance:
    i = 0
    def __init__(self, fq, fd, params, loss_id, pop_id, xfake,xreal,orig_real,cio,uni_roc,bi_roc,im_parent = False):
        self.fq = fq
        self.fd = fd
        self.params = params
        self.pop_id = pop_id
        self.loss_id = loss_id
        self.xfake = xfake
        self.xreal = xreal
        self.im_parent = im_parent
        self.orig_real = orig_real
        self.cio = cio
        self.uni_roc = uni_roc
        self.bi_roc = bi_roc
        self.index = Instance.i
        Instance.i+=1

    def f(self):
        return self.fd

def build_log_template(popsize, nloss):
    #header 
    header = "id"
    header += "\ttime"
    header += "\tmean fake"
    header += "\treal fake"
    header += '\t' + "\t".join(["fake[{}]".format(i) for i in range(popsize)])
    header += '\t' + "\t".join(["real[{}]".format(i) for i in range(popsize)])
    header += '\t' + "\t".join(["fd[{}]".format(i) for i in range(popsize)])
    header += '\t' + "\t".join(["loss[{}]".format(i) for i in range(nloss)])
    #row template
    template = "{}" #id update
    template+= "\t{}" #time
    #template+= "\t{}" #mean fake
    #template+= "\t{}" #mean real
    #template+= "\t{}" * popsize #id update, fake_rate
    #template+= "\t{}" * popsize #id update, real_rate
    template+= "\t{}" * popsize #id update, fd score
    template+= "\t{}" * nloss #id update, fd score
    return header,template

problem_table ={
    "UK" : [
        lambda *args : TaskSynthetic("UK", 300, *args),
        {
           "lr" : 0.0002, 
           "lrd" : 0.0002, 
           "b1" : 0.5,
           'beta' : 0.002,
           "dim" : 128,
           "metric_samples" : 1024
        }
    ],
}

def get_data_on_cond(condvec,sampler,batch_size,fakez):
    if condvec is None:
        c1, m1, col, opt = None, None, None, None
        real,orig_real = sampler.sample_data(batch_size, col, opt)
        return real,fakez,c1,m1,col,opt,None
    else:
        c1, m1, col, opt = condvec
        c1 = torch.from_numpy(c1).cuda()
        m1 = torch.from_numpy(m1).cuda()
        fakez = torch.cat([fakez, c1], dim=1)

        perm = np.arange(batch_size)
        np.random.shuffle(perm)
        real,orig_real = sampler.sample_data(
            batch_size, col[perm], opt[perm])
        c2 = c1[perm]
        return real,fakez,c1,m1,col,opt,c2,orig_real

def main(problem, 
         popsize,
         algorithm, 
         save_freq, 
         loss_type = ['trickLogD','minimax', 'ls'],
         postfix = None,
         nPassD = 1, #backpropagation pass for discriminator
         batchSize = 64,
         metric = "default",
         output_dir="runs",
         gradients_penalty = False,
         graph_update=50,
         log_frequency=True,
         pac=10
         ):
    seed = 42
    if not( problem in problem_table.keys() ):
        exit(-1)
    set_seed(seed)
    # load the datset
    if problem=='UK':
        data_path = '../UKDA-7210-spss/spss/'
        df = pd.read_spss('/content/drive/MyDrive/Aresearch/UKDA-7210-spss/spss/gb91ind.sav')
        df = df.loc[df['REGIONP']=='West Midlands']
        df = df[['AREAP','AGE','COBIRTH','ECONPRIM','ETHGROUP','FAMTYPE','HOURS','LTILL','MSTATUS','QUALNUM',
         'RELAT','SEX','SOCLASS','TRANWORK','TENURE']]
        cate_cols = []
        for col in df.columns:
            if df[col].dtype=='category':
                cate_cols.append(col)
                df[col] = df[col].astype('str')
        print(f'there are {len(cate_cols)} category cols')
        #assert nPassD==1 ############
    print('dataset size:',df.shape)
                
    #X_train,X_test =train_test_split(df,shuffle=True,random_state=42,test_size=0.1)
    #task
    start = time.time()
    transformer = DataTransformer()
    transformer.fit(df, cate_cols)
    dataset = transformer.transform(df)
    sampler = DataSampler(dataset,df,
           transformer.output_info_list,
           log_frequency)
    print(f'fit data after {time.time()-start}s')
    data_dim = transformer.output_dimensions
    task_args = problem_table[problem][1]
    #dataset = {'X_train':X_train,'X_test':X_test}
    task = problem_table[problem][0](nPassD, popsize, batchSize,dataset,sampler,
      transformer,task_args['dim'] , metric,df)
    #net_otype = task.net_output_type() 

    # description
    description_name = '{}_{}_{}_{}'.format( 
        str(task),
        algorithm, 
        popsize,
        postfix if postfix is not None else "",
    ) 
    #build dirs
    
    # share params
    nloss = len(loss_type)
    print(task_args)
    lr  = task_args['lr']   # initial learning rate for adam G
    lrd = task_args['lrd']  # initial learning rate for adam D
    b1 = task_args['b1']    # momentum term of adam
    beta = task_args['beta']  # momentum term of adam
    samples = task_args['metric_samples'] # metric samples
    DIM = task_args['dim']    # momentum term of adam
    GP_norm = gradients_penalty  # if use gradients penalty on discriminator
    LAMBDA = 2.               # hyperparameter sudof GP

    
    def create_generator_trainer(transformer, discriminator, lr=0.0002, b1=0.5, DIM=64):
        return GeneratorTrainer(transformer, task.create_geneator().cuda(), discriminator, lr,  b1)   
    #Fd_auto_normalization = AutoNormalization(float(0.1)) 
    generator_trainer =  create_generator_trainer(transformer,None, lr, b1, DIM)
    #instances = ['/content/drive/MyDrive/runs/UK_smoegan_4_diagrams/models/gen_best_3.pth',
    #             '/content/drive/MyDrive/runs/UK_smoegan_4_diagrams/models/gen_best_1.pth',
    #             '/content/drive/MyDrive/runs/UK_smoegan_4_diagrams/models/gen_best_2.pth',
    #             '/content/drive/MyDrive/runs/UK_smoegan_4_diagrams/models/gen_best_0.pth',]
    instances = ['/content/drive/MyDrive/runs/UK_smoegan_2_mo_fqd/models/last/gen_last_0.pth',
               '/content/drive/MyDrive/runs/UK_smoegan_2_mo_fqd/models/last/gen_last_1.pth',]
    instances =['/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan/last_ctgan.pth',
               '/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan/last_ctgan_1.pth',
               '/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan/last_ctgan_2.pth',
               '/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan/last_ctgan_3.pth',
               '/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan/last_ctgan_4.pth']
    #instances = ['/content/drive/MyDrive/runs/UK_smoegan_2_select8_loss1/models/gen_best_0.pth',
    #           '/content/drive/MyDrive/runs/UK_smoegan_2_select8_loss1/models/gen_best_1.pth',]
    #instances = ['/content/drive/MyDrive/runs/UK_smoegan_2_select8_loss1/models/last/gen_last_0.pth',
    #           '/content/drive/MyDrive/runs/UK_smoegan_2_select8_loss1/models/last/gen_last_1.pth',]
    instances = ['/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan/best_ctgan_1.pth',
               '/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan/last_ctgan_1.pth',]
    task.evaluate( instances, generator_trainer)
    
def str2bool(v):
    return v if isinstance(v, bool) else \
           True if v.lower() in ('yes', 'true', 't', 'y', '1') else \
           False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm","-a", choices=["egan","moegan","smoegan"], default="smoegan")
    parser.add_argument("--loss_type","-l", nargs="+", choices=['trickLogD','minimax', 'ls'], default=['trickLogD','minimax', 'ls'])
    parser.add_argument("--problem","-p", choices=[name for name in list(problem_table.keys())],default=list(problem_table.keys())[0])
    parser.add_argument("--population_size","-mu", type=int, default=8)
    parser.add_argument("--save_frequency","-freq", type=int, default=1000)
    parser.add_argument("--post_fix","-pfix", type=str, default=None)
    parser.add_argument("--update_discrminator","-ud", type=int, default=1)
    parser.add_argument("--batch_size","-bs", type=int, default=500)
    parser.add_argument("--metric","-m", choices=['default','is', 'fid'], default='default')
    parser.add_argument("--gradients_penalty","-gp", type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument("--output_dir","-o", type=str, default="runs")
    parser.add_argument("--graph_update", type=int, default=100)
    parser.add_argument("--pac", type=int, default=10)
    arguments = parser.parse_args()
    print("_"*42)
    print('evaluating models')
    print(" "*14+"> ARGUMENTS <")
    for key in arguments.__dict__:
        print(key+":", arguments.__dict__[key])  
    print("_"*42)
    main(problem=arguments.problem,
         popsize=arguments.population_size,
         algorithm=arguments.algorithm,
         save_freq=arguments.save_frequency,
         loss_type=arguments.loss_type,
         postfix=arguments.post_fix,
         nPassD=arguments.update_discrminator,
         batchSize=arguments.batch_size,
         metric=arguments.metric,
         gradients_penalty=arguments.gradients_penalty,
         output_dir=arguments.output_dir,
         graph_update=arguments.graph_update,
         pac=arguments.pac)
