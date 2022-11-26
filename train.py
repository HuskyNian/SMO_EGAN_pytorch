#standard stuff
import os
import argparse
import random
import json
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
#theano utils
from lib.data_utils import shuffle, iter_data

#train utils
from utils.mab import get_output
from utils.trainers import GeneratorTrainer
from utils.nsga2 import nsga_2_pass
from utils.mmd2u import compute_metric_mmd2
from utils.variation import get_varation, get_varation_names
from utils.task import TaskTrainToy
from utils.task import TaskCifar10
from utils.task import TaskFaces
from utils.task import TaskBedrooms
from utils.timer import Timer
from utils.log import Logger
from utils.auto_normalization import AutoNormalization

#plot stuff
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import clear_output

matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
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
    def __init__(self, fq, fd, params, loss_id, pop_id, img_values, im_parent = False):
        self.fq = fq
        self.fd = fd
        self.params = params
        self.pop_id = pop_id
        self.loss_id = loss_id
        self.img = img_values
        self.im_parent = im_parent

    def f(self):
        return self.fq - self.fd

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
    template+= "\t{}" #mean fake
    template+= "\t{}" #mean real
    template+= "\t{}" * popsize #id update, fake_rate
    template+= "\t{}" * popsize #id update, real_rate
    template+= "\t{}" * popsize #id update, fd score
    template+= "\t{}" * nloss #id update, fd score
    return header,template

problem_table ={
    "8G" : [
        lambda *args : TaskTrainToy("8gaussians", 150 * 1000, *args),
        #lambda *args : TaskTrainToy("8gaussians", 15 * 10, *args),
        {
           "lr" : 0.0001, 
           "lrd" : 0.0001, 
           "b1" : 0.5,
           'beta' : 1.0,
           "dim" : 512,
           "metric_samples" : 1024
        }
    ],
    "25G" : [
        lambda *args : TaskTrainToy("25gaussians", 150 * 1000, *args),
        {
           "lr" : 0.0001, 
           "lrd" : 0.0001, 
           "b1" : 0.5,
           'beta' : 1.0,
           "dim" : 512,
           "metric_samples" : 1024
        }
    ],
    "cifar10" : [
        lambda *args : TaskCifar10("cifar10", 100, *args),
        {
           "lr" : 0.0002, 
           "lrd" : 0.0002, 
           "b1" : 0.5,
           'beta' : 0.002,
           "dim" : 128,
           "metric_samples" : 1024
        }
    ],
    "faces" : [
        lambda *args : TaskFaces("faces", 100, *args),
        {
           "lr" : 0.0002, 
           "lrd" : 0.0002, 
           "b1" : 0.5,
           'beta' : 0.002,
           "dim" : 128,
           "metric_samples" : 128
        }
    ],
    "bedrooms" : [
        lambda *args : TaskBedrooms("bedrooms", 100, *args),
        {
           "lr" : 0.0002, 
           "lrd" : 0.0002, 
           "b1" : 0.5,
           'beta' : 0.002,
           "dim" : 128,
           "metric_samples" : 128
        }
    ],
}

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
         graph_update=50
         ):

    if not( problem in problem_table.keys() ):
        exit(-1)

    #task
    task_args = problem_table[problem][1]
    task = problem_table[problem][0](nPassD, popsize, batchSize, metric)
    #net_otype = task.net_output_type() 

    # description
    description_name = '{}_{}_{}_{}'.format( 
        str(task),
        algorithm, 
        popsize,
        postfix if postfix is not None else "",
    ) 
        
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
    def create_generator_trainer(noise=None, discriminator=None, lr=0.0002, b1=0.5, DIM=64):
        return GeneratorTrainer(noise, task.create_geneator(noise, DIM).cuda(), discriminator, lr,  b1)   

    # MODEL D
    print("Building model and compiling functions...")
    b1_d = 0.
    discriminator = task.create_discriminator(DIM, GP_norm)
    discriminator = discriminator.cuda()
    optimizer = torch.optim.Adam(discriminator.parameters(),lr=lrd,betas=(0,0.999))
    # Prepare Theano variables for inputs and targets
    
    
    '''real_imgs = net_otype('real_imgs')
    fake_imgs = net_otype('fake_imgs')
    # Create neural network model
    discriminator = task.create_discriminator(DIM, GP_norm)
    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator, real_imgs)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(discriminator, fake_imgs)
    # Create loss expressions
    discriminator_loss = (lasagne.objectives.binary_crossentropy(real_out, 1) + lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()

    # Gradients penalty norm
    if GP_norm is True:
        alpha = t_rng.uniform((batchSize, 1), low=0., high=1.)
        differences = fake_imgs - real_imgs
        interpolates = real_imgs + (alpha*differences)
        gradients = theano.grad(lasagne.layers.get_output(discriminator, interpolates).sum(), wrt=interpolates)
        slopes = T.sqrt(T.sum(T.sqr(gradients), axis=(1)))
        gradient_penalty = T.mean((slopes-1.)**2)
        D_loss = discriminator_loss + LAMBDA*gradient_penalty
        b1_d = 0.
    else:
        D_loss = discriminator_loss
        b1_d = 0.'''
        
    def train_discriminator(real_imgs,fake_imgs,discriminator,GP_norm,LAMBDA,optimizer):
        if isinstance(fake_imgs,np.ndarray):
            fake_imgs = torch.FloatTensor(fake_imgs)
        else:
            fake_imgs = fake_imgs.detach()
        fake_imgs,real_imgs = fake_imgs.cuda(),real_imgs.cuda()
        #print('GP_norm is',GP_norm)
        with torch.autograd.set_detect_anomaly(True):
            real_out = get_output(discriminator, real_imgs,train=True)
            # Create expression for passing fake data through the discriminator
            fake_out = get_output(discriminator, fake_imgs,train=True)
            # check the output after sigmoid()
            # Create loss expressions
            discriminator_loss = F.binary_cross_entropy(real_out, torch.ones_like(real_out)) +\
                F.binary_cross_entropy(fake_out, torch.zeros_like(fake_out))
            if GP_norm is True:
                B,C,H,W = real_imgs.shape
                alpha = torch.rand((B, 1, 1, 1)).repeat(1, C, H, W).cuda()
                interpolates = real_imgs * alpha + fake_imgs * (1 - alpha)
                interpolates = torch.autograd.Variable(interpolates, requires_grad=True).cuda()
                disc_interpolates = discriminator(interpolates)
                gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                          grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradients = gradients.reshape(gradients.size(0), -1)
                gradient_penalty = torch.mean(((gradients.norm(2, dim=1) - 1) ** 2)) * LAMBDA
                D_loss = discriminator_loss + gradient_penalty
            else:
                D_loss = discriminator_loss
            #print('D_loss:',D_loss)
            optimizer.zero_grad()
            D_loss.backward()
            optimizer.step()
            return D_loss.cpu().detach().numpy()
    
    def dis_fn(real_imgs,fake_imgs,discriminator,beta):
        real_out,fake_out,Fd_score = _dis_fn(real_imgs,fake_imgs,discriminator,beta)
        return fake_out.mean(),Fd_score
    
    def disft_fn(real_imgs,fake_imgs,discriminator,beta):
        real_out,fake_out,Fd_score = _dis_fn(real_imgs,fake_imgs,discriminator,beta)
        return real_out.mean(), fake_out.mean(), \
          (real_out > 0.5).mean(),  (fake_out > 0.5).mean(), Fd_score
    
    def _dis_fn(real_imgs,fake_imgs,discriminator,beta):
        fake_imgs,real_imgs = fake_imgs.cuda(),real_imgs.cuda()
        real_out = get_output(discriminator, real_imgs,train=True)
        fake_out = get_output(discriminator, fake_imgs,train=True)
        discriminator_loss = F.binary_cross_entropy(real_out, torch.ones_like(real_out)) +\
            F.binary_cross_entropy(fake_out, torch.zeros_like(fake_out))
        Fd = torch.autograd.grad(outputs=discriminator_loss,inputs=discriminator.parameters(),
                                 grad_outputs=torch.ones(discriminator_loss.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)
        Fd_score = beta*torch.log(sum(torch.sum(torch.square(x)) for x in Fd))
        discriminator.zero_grad()
        return [x.cpu().detach().numpy() for x in [real_out,fake_out,Fd_score]]
    # Create update expressions for training
    #discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    #lrtd = theano.shared(lasagne.utils.floatX(lrd))
    #updates_d = lasagne.updates.adam(D_loss, discriminator_params, learning_rate=lrtd, beta1=b1_d)
    #lrt = theano.shared(lasagne.utils.floatX(lr))
    # Fd Socre
    #Fd = theano.gradient.grad(discriminator_loss, discriminator_params)
    #Fd_score = beta*T.log(sum(T.sum(T.sqr(x)) for x in Fd))
    # max is ~7.5 for toy dataset and ~0.025 for real ones (it will be updated after 1 iteration, which is likely the worst one)
    
    Fd_auto_normalization = AutoNormalization(float(0.1)) 

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    #train_d = theano.function([real_imgs, fake_imgs],  discriminator_loss, updates=updates_d)    
    
    # Compile another function generating some data
    #dis_fn = theano.function([real_imgs, fake_imgs], [fake_out.mean(), Fd_score])
    #disft_fn = theano.function([real_imgs, fake_imgs], [real_out.mean(), fake_out.mean(), 
    #                                                    (real_out > 0.5).mean(),  (fake_out > 0.5).mean(), 
    #                                                    Fd_score])

    #main MODEL G
    #noise = T.matrix('noise')
    generator_trainer =  create_generator_trainer(None, discriminator, lr, b1, DIM)

    # Finally, launch the training loop.
    print("Starting training...")
    print(description_name)

    #build dirs
    path_front, path_logs, path_models, path_models_last, path_images = build_output_dirs(output_dir,description_name)

    #define a problem instance
    instances = []
    instances_old = []

    #generator of a offspring
    def generate_offsptring(xreal, loss_id, pop_id, inst = None):
        if inst == None:
            newparams = create_generator_trainer(noise=None, discriminator=discriminator, lr=lr, b1=b1, DIM=DIM).get()
            inst = Instance(-float("inf"), float("inf"), newparams, -1, pop_id,  None)
        #init gen
        generator_trainer.set(inst.params)
        #train
        generator_trainer.train(loss_type[loss_id], task.noise_batch())
        #score
        xfake = generator_trainer.gen(task.noise_batch())
        frr_score, fd_score = dis_fn(xreal, xfake,discriminator,beta)
        #new instance
        new_instance = Instance(frr_score, fd_score, generator_trainer.get(), loss_id, pop_id, xfake)
        #save
        instances.append(new_instance)
        #info stuff
        return new_instance
    
    #init varation
    variation = get_varation(VARIATION)(popsize, nloss, generate_offsptring)

    #reval pop with new D
    def reval_pupulation(in_instances):
        #ret
        out_instances = []
        #generates new batches of images for each generator, and then eval these sets by means (new) D
        for inst in in_instances:
            generator_trainer.set(inst.params)
            xfake = generator_trainer.gen(task.noise_batch())
            frr_score, fd_score = dis_fn(xreal_eval, xfake,discriminator,beta)
            out_instances.append(Instance(
                frr_score, 
                fd_score,
                generator_trainer.get(),
                inst.loss_id,
                inst.pop_id,
                xfake,
                im_parent = True
            ))
        return out_instances
    
    #log stuff
    LOG_HEADER, LOG_TEMPLATE = build_log_template(popsize, nloss)
    log = Logger(os.path.join(path_logs, 'logs.tsv'), header=LOG_HEADER.encode())
    timer = Timer()
    losses_counter=[0]*nloss

    hist_dloss = []
    hist_fake_rate = []
    hist_real_rate = []
    _hist_fake_rate = []
    _hist_real_rate = []
    hist_best_is = []
    hist_worst_is = []
    # We iterate over epochs:
    save_freq = save_freq *64 // batchSize
    print("new save frequency:",save_freq)
    for n_updates in tqdm(task.get_range()):
        #get batch
        xmb = task.batch()
        xmb = torch.FloatTensor(xmb)
        #get eval batch
        if xmb.shape[0]==batchSize:
            xreal_eval = xmb
        else:
            xreal_eval = shuffle(xmb)[:batchSize] 
        # initial G cluster
        if MULTI_OBJECTIVE_SELECTION:
            instances_old = reval_pupulation(instances)
        else:
            instances_old = instances
        #reset
        instances = []
        variation.update(instances_old, task.is_last())
        for pop_id in range(0, popsize):
            variation.gen(xreal_eval, instances_old[pop_id] if n_updates else None, pop_id)

        if popsize <= (len(instances)+len(instances_old)):
            if MULTI_OBJECTIVE_SELECTION==True:
                #add parents in the pool
                instances = [*instances_old,*instances]
                #from the orginal code, we have to maximize D(G(X)),
                #Since in NSGA2 performences a minimization,
                #We are going to minimize -D(G(X)),
                #also we want maximize the diversity score,
                #So, we are going to minimize -diversity score (also we wanna normalize that value)
                cromos = { idx:[-float(inst.fq),
                                -float(Fd_auto_normalization(inst.fd))] for idx,inst in enumerate(instances) } # S2
                cromos_idxs = [ idx for idx,_ in enumerate(instances) ]
                finalpop = nsga_2_pass(popsize, cromos, cromos_idxs)
                instances = [instances[p] for p in finalpop]
                with open(os.path.join(path_front,'last.tsv'), 'wb') as ffront:
                    for inst in instances:
                        ffront.write((str(inst.fq) + "\t" + str(inst.fd)).encode())
                        ffront.write("\n".encode())
            elif nloss>1:
                #sort new
                instances.sort(key=lambda inst: inst.f()) #(from the orginal code in github) maximize
                #cut best ones
                instances = instances[len(instances)-popsize:]

        for i in range(0, popsize):
            xreal, xfake = task.statistic_datas(instances[i].img)
            xreal,xfake = torch.FloatTensor(xreal),torch.FloatTensor(xfake)
            tr, fr, trp, frp, fdscore = disft_fn(xreal, xfake,discriminator,beta)
            fake_rate = np.array([fr]) if i == 0 else np.append(fake_rate, fr)
            real_rate = np.array([tr]) if i == 0 else np.append(real_rate, tr)
            fake_rate_p = np.array([frp]) if i == 0 else np.append(fake_rate_p, frp)
            real_rate_p = np.array([trp]) if i == 0 else np.append(real_rate_p, trp)
            FDL = np.array([fdscore]) if i == 0 else np.append(FDL, fdscore)
            losses_counter[instances[i].loss_id] += 1

        # train D
        for xreal, xfake in task.iter_data_discriminator(xmb, instances):
            this_loss = train_discriminator(xreal,xfake,discriminator,GP_norm,LAMBDA,optimizer)
            hist_dloss.append(this_loss)
            

        #show it info
        #print(n_updates, real_rate.mean(), real_rate_p.mean())
        if n_updates%graph_update==0:
            hist_fake_rate.append(np.mean(_hist_fake_rate))
            hist_real_rate.append(np.mean(_hist_real_rate))
            _hist_fake_rate = []
            _hist_real_rate = []
            plt.plot(np.arange(len(hist_dloss)),hist_dloss,label='train discriminator loss')
            plt.legend()
            plt.savefig(os.path.join(path_images,'discriminator_loss.png'))
            plt.close()
            plt.plot(np.arange(len(hist_fake_rate)),hist_fake_rate,label='fake rate mean')
            plt.plot(np.arange(len(hist_real_rate)),hist_real_rate,label='real rate mean')
            plt.legend()
            plt.savefig(os.path.join(path_images,'fake_real_rate.png'))
            plt.close() # save memory
            if len(hist_best_is)>0:
                plt.plot(np.arange(len(hist_best_is)),hist_best_is,label='best is')
                plt.plot(np.arange(len(hist_best_is)),hist_worst_is,label='worst is')
                plt.legend()
                plt.savefig(os.path.join(path_images,'hist_is.png'))
                plt.close() # save memory
        _hist_fake_rate.append(real_rate_p.mean())
        _hist_real_rate.append(real_rate.mean())
        #write logs
        log.writeln(LOG_TEMPLATE.format(n_updates, 
                                        str(timer), 
                                        fake_rate.mean(), 
                                        real_rate.mean(), 
                                        *fake_rate,  
                                        *real_rate, 
                                        *FDL,
                                        *losses_counter).encode())
        #varation logs
        variation.logs(path_logs, n_updates, last_iteration = task.is_last())
        
        if (n_updates % save_freq == 0 and n_updates!=0) or n_updates==1 or task.is_last():
            #it same
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
            metric_results = task.compute_metrics(instances_to_eval, lambda inst, nz: generator_trainer.set(inst.params).gen(nz), samples)
            #mmd2 output
            print(n_updates, "metric:", np.min(metric_results), "id:", np.argmin(metric_results))
            hist_best_is.append(np.min(metric_results))
            hist_worst_is.append(np.max(metric_results))
            content = {"hist_dloss":hist_dloss ,
                            "hist_fake_rate":hist_fake_rate,
                           "hist_real_rate":hist_real_rate,
                            "hist_best_is":hist_best_is,
                            "hist_worst_is":hist_worst_is}
            np.save(os.path.join(path_logs,'hist_logs.txt'),content)

            #best
            best = np.argmin(metric_results)
            worst = np.argmax(metric_results)
            #np.savez(os.path.join(path_models,'dis_%s.npz')%(id_name_update), *lasagne.layers.get_all_param_values(discriminator))
            
            torch.save(discriminator.state_dict(),os.path.join(path_models,'dis_last.pth'))
            torch.save(instances_to_eval[best].params,os.path.join(path_models,'gen_lastbest.pth') )
            #save best
            generator_trainer.set(instances_to_eval[best].params)
            xfake_best = generator_trainer.gen(task.noise_batch(samples))
            #worst_debug
            generator_trainer.set(instances_to_eval[worst].params)
            xfake_worst = generator_trainer.gen(task.noise_batch(samples))
            #save images
            task.save_image(xmb, xfake_best, path_images, "best_%s" % (id_name_update))
            task.save_image(xmb, xfake_worst, path_images, "worst_%s" % (id_name_update))
            #print pareto front
            with open(os.path.join(path_front,'%s.tsv') % (id_name_update), 'wb') as ffront:
                for idx in range(len(instances_to_eval)):
                    ffront.write((str(instances_to_eval[idx].fq) + "\t" + str(instances_to_eval[idx].fd) + "\t" + str(metric_results[idx])).encode())
                    ffront.write("\n".encode())
            #save all last models:
            if task.is_last():
                for key,inst in enumerate(instances_to_eval):
                    torch.save(inst.params,os.path.join(path_models_last,'gen_%s.npz')%(key))
                    


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
    parser.add_argument("--batch_size","-bs", type=int, default=64)
    parser.add_argument("--metric","-m", choices=['default','is', 'fid'], default='default')
    parser.add_argument("--gradients_penalty","-gp", type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument("--output_dir","-o", type=str, default="runs")
    parser.add_argument("--graph_update", type=int, default=100)
    arguments = parser.parse_args()
    print("_"*42)
    print(" "*14+"> ARGUMENTS <")
    for key in arguments.__dict__:
        print(key+":", arguments.__dict__[key])  
    print("_"*42)
    seed = 42
    print("seed: ",seed)
    set_seed(seed)
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
         graph_update=arguments.graph_update)

'''def run(problem, 
         popsize,
         algorithm, 
         save_freq, 
         loss_type = ['trickLogD','minimax', 'ls'],
         postfix = None,
         nPassD = 1, #backpropagation pass for discriminator
         batchSize = 64,
         metric = "default",
         output_dir="runs",
         gradients_penalty = False):
    arguments = parser.parse_args()
    print("_"*42)
    print(" "*14+"> ARGUMENTS <")
    for key in arguments.__dict__:
        print(key+":", arguments.__dict__[key])  
    print("_"*42)
    main(problem=problem,
         popsize=population_size,
         algorithm=algorithm,
         save_freq=save_frequency,
         loss_type=loss_type,
         postfix=post_fix,
         nPassD=update_discrminator,
         batchSize=batch_size,
         metric=metric,
         gradients_penalty=gradients_penalty,
         output_dir=output_dir,
         graph_update=graph_update)'''
    
    
    
'''def main(problem, 
         popsize,
         algorithm, 
         save_freq, 
         loss_type = ['trickLogD','minimax', 'ls'],
         postfix = None,
         nPassD = 1, #backpropagation pass for discriminator
         batchSize = 64,
         metric = "default",
         output_dir="runs",
         gradients_penalty = False
         ):

    if not( problem in problem_table.keys() ):
        exit(-1)

    #task
    task_args = problem_table[problem][1]
    task = problem_table[problem][0](nPassD, popsize, batchSize, metric)
    
"8G" : [
    lambda *args : TaskTrainToy("8gaussians", 150 * 1000, *args),
    {
       "lr" : 0.0001, 
       "lrd" : 0.0001, 
       "b1" : 0.5,
       'beta' : 1.0,
       "dim" : 512,
       "metric_samples" : 1024
    }
],'''
