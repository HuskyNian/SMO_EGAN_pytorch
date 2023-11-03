import os
import random
import numpy as np
import math
import itertools
import json
from utils.mab import UCB1
from utils.mab import QFunction
from utils.mab import DeepQFunction
import torch

class Varation:

    def update(self, instances, last_iteration = False):
        if  instances is None or len(instances) == 0:
            return
        self._update(instances, last_iteration)

    def _update(self, instances, last_iteration = False):
        pass

    def gen(self,samples, instance, instance_id):
        pass

    def logs(self, path_logs, iteration, last_iteration = False):
        pass 

class VarationAll(Varation):

    def __init__(self,  popsize, nloss, generate_offsptring):
        self.generate_offsptring = generate_offsptring
        self.nloss = nloss

    def gen(self, samples, instance, instance_id):        
        for loss_id in range(0,self.nloss):
            self.generate_offsptring(samples, loss_id, instance_id, instance)

class SarsaBatch:
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.next_actions = []
        self.rewards = []

    def append(self, state, action, next_state, next_action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.next_actions.append(next_action)
        self.rewards.append(reward)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.next_actions = []
        self.rewards = []

    def __len__(self):
        return len(self.states)

class VarationDeepQLearning(Varation):
    
    #replace a loss as a arm to a configuration set
    #the reward will be the distance between the new config with le old one

    def __init__(self, popsize, nloss, generate_offsptring):        
        self.generate_offsptring = generate_offsptring
        self.nloss = nloss
        self.popsize = popsize
        self.epsilon = 0.7
        self.epsilon_min = 0.1
        self.epsilon_dec = 0.999
        self.norm_loss = lambda x : float(x) / (self.nloss+1) + 1.0 / (self.nloss+1)
        # ###############################################################################  
        self.fd_min = -1# float("inf")
        self.fd_max =  7# 9 #-float("inf")
        def _encode_state(instance):
            #self.fd_min = min(self.fd_min, instance.fd)
            #self.fd_max = max(self.fd_max, instance.fd)
            #norm_fd = (instance.fd - self.fd_min) / (self.fd_max - self.fd_min)
            return (instance.fq,  instance.fd)  #### risk should be normalized to 0-1, now it's between -1~1
        self.encode_state = _encode_state
        self.state_size = 2 #3  # input dim of the neural networ for action choosing
        # ###############################################################################  
        self.batch = SarsaBatch()
        self.batch_size = 32
        self.first_epoch = True
        self.learning_rate = 0.1 #0.1 #/ self.batch_size
        self.qfun = DeepQFunction(size=self.state_size,
                                  nactions=nloss,
                                  discount=0.925,
                                  learning_rate=self.learning_rate,
                                  use_sarsa=True)
        self.info = [None for _ in range(popsize)]
        self.actions_chosen = [0]*self.nloss

    def chouse_action(self, state, force_random = False):
        if random.random() <= self.epsilon or force_random:
            return random.randint(0, self.nloss - 1)
        return self.qfun.action(state)

    def update_epsilon(self):
        self.epsilon *= self.epsilon_dec
        self.epsilon  = max(self.epsilon_min, self.epsilon) 

    def _update(self, instances, last_iteration = False):
        #update
        if len(self.batch) >= self.batch_size:
            self.qfun.update_sarsa(self.batch.states, 
                                   self.batch.actions,
                                   self.batch.next_states,
                                   self.batch.next_actions,
                                   self.batch.rewards)
            self.batch.clear()
            self.first_epoch = False
        #update epsilon
        if not self.first_epoch:
            self.update_epsilon()

    def gen(self,  instance, instance_id,instances,generator_trainer,metric=True): 
        #choise
        if instance is not None:
            state = self.encode_state(instance)
            action = self.chouse_action(state, force_random = self.first_epoch)
            self.info[instance_id] = (state, action)
        else:
            action = random.randint(0, self.nloss - 1)
        #debug
        self.actions_chosen[action] += 1
        #gen
        new_instance,cond_loss,g_loss = self.generate_offsptring( action, instance_id,instances,generator_trainer, instance,metric=metric)
        #add into batch
        if instance is not None:
            next_state = self.encode_state(new_instance)
            
            #add into batch
            utility_before = state[1]
            utility_after = next_state[1]
            if utility_after!=utility_before:
                next_action = self.qfun.action(state)
                #print('utility_before and after',utility_before,utility_after)
                self.batch.append(
                    state,
                    action,
                    next_state,
                    next_action,
                    int(utility_after > utility_before)
                )
                # reward is 1 if the new utility score is large than before
        return cond_loss,g_loss
        


    def logs(self, path_logs, iteration, last_iteration = False):
        if iteration == 0:
            with open(os.path.join(path_logs, "qlearning.tsv"),"w") as flog:
                flog.write("{}\t".format("Gen"))
                flog.write("\t".join(["Action_"+str(i) for i in range(self.nloss)]))
                flog.write("\n")
        #for each update
        with open(os.path.join(path_logs, "qlearning.tsv"),"a") as flog:
            flog.write("{}\t".format(iteration))
            flog.write("\t".join([str(v) for v in self.actions_chosen]))
            flog.write("\n")
        #save net
        if iteration > 0:
            if last_iteration or (iteration % 1000) == 0: ### 
                torch.save(self.qfun.params(),os.path.join(path_logs,'gen_%s.pth')%(int(iteration /  1000)) )




varation_factory = {
   'all'      : VarationAll,
   "deepqlearning": VarationDeepQLearning
}

def get_varation(variation_type):
    if variation_type in varation_factory:
        return varation_factory[variation_type]
    else:
        return Varation

def get_varation_names():
    return [name for name in varation_factory]
        