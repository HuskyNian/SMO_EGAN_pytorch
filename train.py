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
import pickle
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
    "Canada" : [
        lambda *args : TaskSynthetic("Canada", 60, *args),
        {
           "lr" : 0.0002, 
           "lrd" : 0.0002, 
           "b1" : 0.5,
           'beta' : 0.002,
           "dim" : 128,
           "metric_samples" : 1024
        }
    ],
    "Fiji" : [
        lambda *args : TaskSynthetic("Fiji", 60, *args),
        {
           "lr" : 0.0002, 
           "lrd" : 0.0002, 
           "b1" : 0.5,
           'beta' : 0.002,
           "dim" : 128,
           "metric_samples" : 1024
        }
    ],
    "Rwanda" : [
        lambda *args : TaskSynthetic("Rwanda", 300, *args),
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
         pac=10,arguments=None
         ):
    seed = arguments.seed
    if not( problem in problem_table.keys() ):
        exit(-1)
    set_seed(seed)
    # load the datset
    if problem=='UK':
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
    elif problem=='Canada':
        df = pd.read_csv('/content/drive/MyDrive/Aresearch/census/Canada2011_census_microdata.csv')
        df.columns = [i.replace('CA2011A_','') for i in df.columns] 
        cate_cols = []
        c_cols = ['HRSWK','INCTOT','WKSWORK']
        for col in df.columns:
            if col not in c_cols:
                cate_cols.append(col)
        print(f'there are {len(cate_cols)} category cols')
    elif problem=='Fiji':
        df = pd.read_csv('/content/drive/MyDrive/Aresearch/census/Fiji2007_census_microdata.csv')
        df.columns = [i.replace('FJ2007A_','') for i in df.columns] 
        cate_cols = []
        c_cols = []
        for col in df.columns:
            if col not in c_cols:
                cate_cols.append(col)
        print(f'there are {len(cate_cols)} category cols')
    elif problem=='Rwanda':
        df = pd.read_csv('/content/drive/MyDrive/Aresearch/census/Rwanda2012_census_microdata.csv')
        df.columns = [i.replace('RW2012A_','') for i in df.columns] 
        cate_cols = []
        c_cols = []
        for col in df.columns:
            if col not in c_cols:
                cate_cols.append(col)
        print(f'there are {len(cate_cols)} category cols')

      
    print('dataset size:',df.shape)
                
    #X_train,X_test =train_test_split(df,shuffle=True,random_state=42,test_size=0.1)
    #task
    start = time.time()
    transformer = DataTransformer()
    transformer.fit(df, cate_cols)
    dataset = transformer.transform(df)
    '''print(transformer.output_info_list)
    print(transformer._column_raw_dtypes)
    print(transformer.output_dimensions)
    print(transformer._column_transform_info_list)'''

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
    description_name = '{}_{}_{}_{}{}'.format( 
        str(task),
        algorithm, 
        popsize,
        postfix if postfix is not None else "",
        str(seed)
    ) 
    #build dirs
    path_front, path_logs, path_models, path_models_last, path_images = build_output_dirs(output_dir,description_name)
    print('path front, logs, models, models_last, images',path_front, path_logs, path_models, path_models_last, path_images)

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

    # algorithm params
    if algorithm == "egan":
        VARIATION = "all"
        MULTI_OBJECTIVE_SELECTION = False
    elif algorithm == "moegan":
        VARIATION = "all"
        MULTI_OBJECTIVE_SELECTION = True
    elif algorithm == "smoegan":
        VARIATION = "deepqlearning"
        MULTI_OBJECTIVE_SELECTION = True
    else:
        exit(-2)
    # Load the dataset
    

    # MODEL D
    print("Building model and compiling functions...")
    discriminator = task.create_discriminator().cuda()
    optimizer = torch.optim.Adam(discriminator.parameters(),lr=lrd,betas=(b1,0.9),weight_decay=1e-6)
        
    def _dis_fn(real_imgs,fake_imgs,discriminator,beta):
        if isinstance(fake_imgs,np.ndarray):
            fake_imgs = torch.tensor(fake_imgs)
        if isinstance(real_imgs,np.ndarray):
            real_imgs = torch.tensor(real_imgs)   
        with torch.no_grad():
            fake_imgs,real_imgs = fake_imgs.cuda(),real_imgs.cuda()
            real_out = discriminator(real_imgs)
            fake_out = discriminator(fake_imgs)
        
        return real_out.cpu().detach().numpy(),fake_out.cpu().detach().numpy()

    def dis_fn(real_imgs,fake_imgs,orig_real):
        xfake = task.transformer.inverse_transform(fake_imgs.cpu().detach().numpy())
        start = time.time()
        tcap = cal_mean_tcap(task.name,orig_real,xfake)
        #print('roc fun:',time.time()-start)
        start = time.time()
        cio = cal_mean_cio(task.name,orig_real,xfake,task.ordinal_encoder)
        start = time.time()
        uni_roc,bi_roc = cal_mean_roc(task.name,orig_real,xfake,bi=False)
        utility = (cio+uni_roc+bi_roc)/3
        return tcap,utility,cio,uni_roc,bi_roc
    
    def disft_fn(real_imgs,fake_imgs,discriminator,beta):
        real_out,fake_out= _dis_fn(real_imgs,fake_imgs,discriminator,beta)
        return real_out.mean(), fake_out.mean(), \
          (real_out > 0.5).mean(),  (fake_out > 0.5).mean()
    
    def create_generator_trainer(transformer, discriminator, lr=0.0002, b1=0.5, DIM=64):
        return GeneratorTrainer(transformer, task.create_geneator().cuda(), discriminator, lr,  b1)   
    #Fd_auto_normalization = AutoNormalization(float(0.1)) 
    generator_trainer =  create_generator_trainer(transformer, discriminator, lr, b1, DIM)

    # Finally, launch the training loop.
    print("Starting training...")
    print(description_name)

    #define a problem instance
    instances = []
    instances_old = []
    def train_discriminator(real,fakeact,
                            discriminator,GP_norm,LAMBDA,optimizer):
        if isinstance(fakeact,np.ndarray):
            fakeact = torch.FloatTensor(fakeact)
        fakeact,real = fakeact.detach().cuda(),real.detach().cuda()
        real_cat = real  # already concat the conditional vector
        fake_cat = fakeact
        
        #print('d optimizer 3 times')
        real_out = discriminator(real_cat)
        # Create expression for passing fake data through the discriminator
        fake_out = discriminator(fake_cat)
        #print('GP_norm is',GP_norm)
        # check the output after sigmoid()
        # Create loss expressions
        #discriminator_loss = -(torch.mean(real_out)-torch.mean(fake_out))
        with torch.autograd.set_detect_anomaly(True):
            pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, 'cuda', pac)
            discriminator_loss = F.binary_cross_entropy(real_out, torch.ones_like(real_out)) +\
                F.binary_cross_entropy(fake_out, torch.zeros_like(fake_out))
            #discriminator_loss = -(torch.mean(real_out) - torch.mean(fake_out)) ################
            optimizer.zero_grad()
            pen.backward(retain_graph=True)
            discriminator_loss.backward()
            optimizer.step()
        return discriminator_loss.cpu().detach().numpy()
    #generator of a offspring
    hist_cond_losses = []
    hist_g_losses = []
    hist_dloss = []
    def generate_offsptring( loss_id, pop_id,instances,generator_trainer,inst = None,metric=True):
        if True:
            if inst != None:
                generator_trainer.set(inst.params)
            condvec = sampler.sample_condvec(batchSize) 
            noise = task.get_fakez(batchSize)
            xreal,fakez,c1,m1,col,opt,c2,orig_real =  get_data_on_cond(condvec,sampler,batchSize,noise)
            #train
            #print(f'training g with loss{loss_type[loss_id]}, popid:{pop_id}')
            train_fake,cond_loss,loss = generator_trainer.train(loss_type[loss_id], (fakez,c1,m1)) # generate at train()

            #score
            condvec = sampler.sample_condvec(batchSize) 
            noise = task.get_fakez(batchSize)
            xreal,fakez,c1,m1,col,opt,c2,orig_real =  get_data_on_cond(condvec,sampler,batchSize,noise)
            xreal = torch.FloatTensor(xreal).cuda()
            xfake = generator_trainer.gen(fakez) # generate at train(), after activation
            if c1 is not None:
                xfake= torch.cat([xfake, c1], dim=1)
                xreal = torch.cat([xreal, c2], dim=1)
            if metric or inst is None:
                frr_score, fd_score,cio,uni_roc,bi_roc = dis_fn(xreal, xfake,orig_real)#############
                #frr_score, fd_score,cio,uni_roc,bi_roc =1,1,1,1,1
                #new instance
                new_instance = Instance(frr_score, fd_score, generator_trainer.get(), 
                                        loss_id, pop_id, xfake,xreal,orig_real,cio=cio,
                                        uni_roc=uni_roc,bi_roc=bi_roc)
            else:
                new_instance = Instance(inst.fq, inst.fd, generator_trainer.get(), 
                                        loss_id, pop_id, xfake,xreal,orig_real,
                                        cio=inst.cio,uni_roc=inst.uni_roc,bi_roc=inst.bi_roc)
                inst = new_instance
            instances.append(new_instance)
            return new_instance,cond_loss,loss
        else:
            instances.append(inst)
            return inst,0,0

    #init varation
    variation = get_varation(VARIATION)(popsize, nloss, generate_offsptring)

    
    #reval pop with new D
    
    #log stuff
    LOG_HEADER, LOG_TEMPLATE = build_log_template(popsize, nloss)
    log = Logger(os.path.join(path_logs, 'logs.tsv'), header=LOG_HEADER.encode())
    timer = Timer()
    losses_counter=[0]*nloss

    
    hist_fake_rate = []
    hist_real_rate = []
    _hist_fake_rate = []
    _hist_real_rate = []
    hist_utility = []
    hist_risk = []
    best_utility =0
    best_risk = 0
    epoch = 0
    # We iterate over epochs:
    save_freq = len(df) // batchSize
    select_freq = arguments.select
    print("new save frequency:",save_freq)
    with open(os.path.join(path_front,'pareto_fronts.csv'),'w',encoding='utf-8') as f:
        f.write('epoch,tcap,utility,cio,uni_roc,bi_roc,\n')
    for n_updates in tqdm(task.get_range()):
        # most important is conditional vector, real and fake data and fair 
        # comparison are based on that
        #get eval batch
        # initial G cluster
        #print_g(instances,old=True)
        for ins in instances:
            ins.im_parent = True
        instances_old = instances
        #reset
        instances = []
        if nloss>1:
            variation.update(instances_old, task.is_last())
        start = time.time()
        if n_updates%select_freq==0:
            for pop_id in range(0, popsize):
                cond_loss,g_loss = variation.gen( instances_old[pop_id] if n_updates else None, pop_id,instances,generator_trainer)
        else:
            for pop_id in range(0, popsize): 
                cond_loss,g_loss = variation.gen( instances_old[pop_id] if n_updates else None, pop_id,instances,generator_trainer,metric=False)

        if popsize <= (len(instances)+len(instances_old)) and n_updates%select_freq==0: 
            if MULTI_OBJECTIVE_SELECTION==True:
                #print('multi object selection')
                #add parents in the pool
                instances = [*instances_old,*instances]
                #from the orginal code, we have to maximize D(G(X)),
                #Since in NSGA2 performences a minimization,
                #We are going to minimize -D(G(X)),
                #also we want maximize the diversity score,
                #So, we are going to minimize -diversity score (also we wanna normalize that value)
                #cromos = { idx:[-float(inst.fq),
                #                -float(Fd_auto_normalization(inst.fd))] for idx,inst in enumerate(instances) } # S2
                
                cromos = { idx:[max(inst.fq,0.),   # here fq is risk, which is to be minimize, so no need to add -
                                -inst.fd] for idx,inst in enumerate(instances) } #
                #cromos = { idx:[-inst.cio,   # here fq is risk, which is to be minimize, so no need to add -
                #                -(inst.bi_roc+inst.uni_roc)/2] for idx,inst in enumerate(instances) } #
                #here fd is utility, which is to be maximized, so add -, already in 0-1, so no normalization
                
                
                cromos_idxs = [ idx for idx,_ in enumerate(instances) ]
                finalpop = nsga_2_pass(popsize, cromos, cromos_idxs)
                instances = [instances[p] for p in finalpop]
                #with open(os.path.join(path_front,'last.tsv'), 'wb') as ffront:
                #    for inst in instances:
                #        ffront.write((str(inst.fq) + "\t" + str(inst.fd)).encode())
                #        ffront.write("\n".encode())

                fqs = [instances[idx].fq for idx in range(len(instances))]
                fds = [instances[idx].fd for idx in range(len(instances))]
                log_df = pd.DataFrame({'tcap':fqs,  'utility':fds})
                log_df.to_csv(os.path.join(path_front,'last.csv') )
            elif nloss>1:
                print('sort sort')
                #sort new
                instances.sort(key=lambda inst: inst.f()) #(from the orginal code in github) maximize
                #cut best ones
                instances = instances[len(instances)-popsize:]
        start = time.time()
        for i in range(0, popsize):
            #xreal, xfake = instances[i].xreal,instances[i].xfake
            #xreal,xfake = xreal.cuda(),xfake.cuda()
            #tr, fr, trp, frp = disft_fn(xreal, xfake,discriminator,beta)
            #fake_rate = np.array([fr]) if i == 0 else np.append(fake_rate, fr)
            #real_rate = np.array([tr]) if i == 0 else np.append(real_rate, tr)
            #fake_rate_p = np.array([frp]) if i == 0 else np.append(fake_rate_p, frp)
            #real_rate_p = np.array([trp]) if i == 0 else np.append(real_rate_p, trp)
            FDL = np.array([instances[i].fd]) if i == 0 else np.append(FDL, instances[i].fd)
            losses_counter[instances[i].loss_id] += 1
        start = time.time()
        # train D
        for inst in instances:  # reassign new training data for a parent
            if inst.im_parent:
                #print('assign samples to parents')
                generator_trainer.set(inst.params)
                condvec = sampler.sample_condvec(batchSize) 
                noise = task.get_fakez(batchSize)
                xreal,fakez,c1,m1,col,opt,c2,orig_real =  get_data_on_cond(condvec,sampler,batchSize,noise)
                xreal = torch.FloatTensor(xreal).cuda()
                xfake = generator_trainer.gen(fakez) # generate at eval(), after activation
                if c1 is not None:
                    xfake= torch.cat([xfake, c1], dim=1)
                    xreal = torch.cat([xreal, c2], dim=1)
                inst.xfake = xfake
                inst.xreal = xreal
        for xreal, xfake in task.iter_data_discriminator( instances):
            #print('training d, xshape',xreal.shape)
            this_loss = train_discriminator(xreal,xfake,discriminator,GP_norm,LAMBDA,optimizer)
        if n_updates%10==0: #log D loss
            hist_dloss.append(this_loss)
            hist_cond_losses.append(cond_loss)
            hist_g_losses.append(g_loss)
            
        start = time.time()
        #show it info
        #print(n_updates, real_rate.mean(), real_rate_p.mean())
        if n_updates%graph_update==0: 
            plt.plot(np.arange(len(hist_g_losses)),hist_g_losses,label='generator loss')
            plt.plot(np.arange(len(hist_g_losses)),hist_cond_losses,label='generator condition loss')
            plt.xlabel('training steps (*10)')
            plt.legend()
            plt.savefig(os.path.join(path_images,f'hist_g_loss.png'))
            plt.close()
            plt.plot(np.arange(len(hist_g_losses)),hist_dloss,label='discriminator loss')
            plt.xlabel('training steps (*10)')
            plt.legend()
            plt.savefig(os.path.join(path_images,f'hist_d_loss.png'))
            plt.close()
            #hist_fake_rate.append(np.mean(_hist_fake_rate))
            #hist_real_rate.append(np.mean(_hist_real_rate))
            #_hist_fake_rate = []
            #_hist_real_rate = []
            #plt.plot(np.arange(len(hist_fake_rate)),hist_fake_rate,label='fake rate mean')
            #plt.plot(np.arange(len(hist_real_rate)),hist_real_rate,label='real rate mean')
            #plt.legend()
            #plt.savefig(os.path.join(path_images,'fake_real_rate.png'))
            #plt.close() # save memory
            if len(hist_risk)>0:
                plt.plot(np.arange(len(hist_utility)),hist_utility,label='utility')
                plt.plot(np.arange(len(hist_risk)),hist_risk,label='risk')
                plt.xlabel('epochs')
                plt.legend()
                plt.savefig(os.path.join(path_images,'hist_utility_risk.png'))
                plt.close() # save memory
        #_hist_fake_rate.append(real_rate_p.mean())
        #_hist_real_rate.append(real_rate.mean())
        #write logs
        log.writeln(LOG_TEMPLATE.format(n_updates, 
                                        str(timer), 
                                        #fake_rate.mean(), 
                                        #real_rate.mean(), 
                                        #*fake_rate,  
                                        #*real_rate, 
                                        *FDL,
                                        *losses_counter).encode())
        #varation logs
        variation.logs(path_logs, n_updates, last_iteration = task.is_last())
        
        if (n_updates % save_freq == 0 and n_updates!=0) or n_updates==1 or task.is_last(): 
            #it same
            #print('eval and save model at updates:',n_updates)
            if task.is_last():
                id_name_update = math.ceil(float(n_updates)/save_freq)
            else:
                id_name_update = math.floor(float(n_updates)/save_freq)
            #if is egan, eval only the best one.
            if MULTI_OBJECTIVE_SELECTION==True:
                instances_to_eval = instances
            else:
                instances_to_eval = [instances[-1]]
            #metric
            metric_results = task.compute_metrics(instances_to_eval,  generator_trainer, samples)
            #mmd2 output
            
            content = {"hist_dloss":hist_dloss ,
                            "hist_utility":hist_utility,
                            "hist_risk":hist_risk}
            np.save(os.path.join(path_logs,'hist_logs.txt'),content)

            torch.save(discriminator.state_dict(),os.path.join(path_models,'dis_last.pth'))

            util_all = (metric_results['uni_roc']+metric_results['bi_roc']+ metric_results['cio'])/3
            max_i = np.argmax(util_all)
            overall_utility = util_all[max_i]
            #overall_utility = (metric_results['uni_roc'].mean()+metric_results['bi_roc'].mean() + metric_results['cio'].mean())/3
            #risk = metric_results['tcap'].mean()
            risk = metric_results['tcap'][max_i]
            #print(n_updates, "overall utility:", overall_utility,'risk',risk,
            #'cio:',metric_results['cio'][max_i],'uni_roc:',metric_results['uni_roc'][max_i],
            #'bi_roc:',metric_results['bi_roc'][max_i])
            print(n_updates,f'epoch:{epoch}', "overall utility:", overall_utility,'risk',risk,
            'cio:',metric_results['cio'],'uni_roc:',metric_results['uni_roc'],
            'bi_roc:',metric_results['bi_roc'])
            
            hist_utility.append(overall_utility)
            hist_risk.append(risk)

            is_best = (overall_utility-best_utility)*2 + max(best_risk,0) - max(risk,0)
            if overall_utility>best_utility :
                #if is_best>=0:
                best_utility = overall_utility
                best_risk = risk
                for model_index in range(len(instances_to_eval)):
                    torch.save(instances_to_eval[model_index].params,os.path.join(path_models,f'gen_best{epoch}_{model_index}.pth') )
            print(f'best utility: {best_utility} risk:{best_risk}')
            if n_updates % (save_freq*25) == 0:
                for model_index in range(len(instances_to_eval)):
                    torch.save(instances_to_eval[model_index].params,os.path.join(path_models_last,f'gen_last_{model_index}.pth') )
            
            #print pareto front
            fqs = [instances_to_eval[idx].fq for idx in range(len(instances_to_eval))]
            fds = [instances_to_eval[idx].fd for idx in range(len(instances_to_eval))]
            log_df = pd.DataFrame({'tcap':fqs,  'utility':fds,  'uni_roc':metric_results['uni_roc'],  'bi_roc':metric_results['bi_roc'], 
             'cio':metric_results['cio']})
            #log_df.to_csv(os.path.join(path_front,'%s.csv') % (id_name_update))
            with open(os.path.join(path_front,'pareto_fronts.csv'),'a',encoding='utf-8') as f:
                for i in range(len(instances_to_eval)):
                    f.write(f'{epoch},{metric_results["tcap"][i]},\
                    {util_all[i]},{metric_results["cio"][i]},\
                    {metric_results["uni_roc"][i]},{metric_results["bi_roc"][i]},\n')
            epoch+=1

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select", type=int, default=8) # select frequency
    arguments = parser.parse_args()
    print("_"*42)
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
         pac=arguments.pac,arguments = arguments)
