import math
import random
import numpy as np 
import torch
import torch.nn as nn
class UCB1():

    def __init__(self, counts = None, 
                       values = None,
                       n_arms = None,
                       reward_interpolation = None):
        if n_arms is not None:
            self.initialize(n_arms)
        else:
            self.counts = counts # Count represent counts of pulls for each arm. For multiple arms, this will be a list of counts.
            self.values = values # Value represent average reward for specific arm. For multiple arms, this will be a list of values.
        #how work in the time
        if reward_interpolation is None:
            self.interpolation = lambda value,reward,n : ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        else:
            self.interpolation = reward_interpolation
    
    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return
    
    def get_prefered_arm(self):
        return self.values.index(max(self.values))
    
    def get_arms_values(self):
        return self.values

    # UCB arm selection based on max of UCB reward of each arm
    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm
    
        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)

        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus

        return ucb_values.index(max(ucb_values))
    
    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        # Update average/mean value/reward for chosen arm
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = self.interpolation(value, reward, n)


class QTable():

    def __init__(self, nactions):
        self.table = {}
        self.nactions = nactions

    def __getitem__(self, state):
        if state in self.table:
            return self.table[state]
        else: 
            self.table[state] = [0.0 for a in range(self.nactions)]
            return self.table[state]
    
    def action(self,state):
        actions = self.__getitem__(state)
        if sum(actions) == 0.0:
            return random.randint(0, self.nactions-1)
        else:
            return np.argmax(actions)

class QFunction():

    def __init__(self, nactions, discount=1.0, learning_rate=0.1):
        self.nactions = nactions
        self.discount = discount
        self.learning_rate = learning_rate
        self.table = QTable(nactions)

    #TIME-DIFFERENCE SUPERVISED (or SARSA)
    def update_sarsa(self, state, action, next_state, next_action, reward):
        qvalue = self.table[state][action]
        next_qvalue = self.table[next_state][next_action]
        new_qvalue = qvalue + self.learning_rate * ( reward + self.discount * next_qvalue - qvalue)
        self.table[state][action] = new_qvalue

    def update(self, state, action, next_state, reward):
        qvalue = self.table[state][action]
        next_qvalue = np.max(self.table[next_state])
        new_qvalue = qvalue + self.learning_rate * ( reward + self.discount * next_qvalue - qvalue)
        self.table[state][action] = new_qvalue

    def action(self, state):
        return self.table.action(state)
    
def init_weights(module):
    if isinstance(module,(nn.Conv2d,nn.Linear,nn.ConvTranspose2d)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
class Network(nn.Module):
    def __init__(self,size,actions,nd=30):
        super().__init__()
        act = nn.ReLU()
        self.layers = nn.Sequential(nn.Linear(size,nd),act,
                                    nn.Linear(nd,nd),act,
                                    nn.Linear(nd,nd),act,
                                    nn.Linear(nd,actions),nn.Softmax(dim=1)
                                    )
        self.apply(init_weights)
    def forward(self,x):
        x = self.layers(x)
        return x

def get_output(network,inputs,train):
    if train:
        network.train()
        return network(inputs)
    else:
        with torch.no_grad():
            network.eval()
            out = network(inputs)
        return out

class DeepQFunction():

    def __init__(self, size, nactions, discount=1.0, learning_rate=0.1, use_sarsa=False):
        self.nactions = nactions
        self.discount = discount
        self.learning_rate = learning_rate
        #self.input = T.matrix("input")
        self.size = size
        self.network = DeepQFunction.build_network(size, nactions)
        self._updates_net = torch.optim.Adam(self.network.parameters(),lr=self.learning_rate,
                                             betas=(0., 0.999))
        self.qfunction = get_output
        if use_sarsa:
            self.qtrain = self.build_training_sarsa_function
        else:
            self.qtrain = self.build_training_function  

    @staticmethod
    def build_network(size, actions, nd = 30):
        net = Network(size,actions,nd)
        return net
    
    def action(self,state):
        state = torch.FloatTensor(np.array(state)).reshape((1, self.size))  #as f32 matrix
        return np.argmax(self.qfunction(self.network,state,train=False))
    
    def get_loss_function(self,states,rewards,next_states,actions):
        #self.states = T.matrix('state')
        #self.rewards = T.col('reward')
        #self.next_states = T.matrix('next_state')        
        #self.actions = T.icol('action')        
        #q(s,a)
    
        actionmask = torch.eq(torch.arange(self.nactions).reshape((1, -1)), actions.reshape((-1, 1)))
        actionmask = actionmask.float()
        q_action = (get_output(self.network, states,train=True) * actionmask).sum(axis=1).reshape((-1, 1))
        #max(q(s_next))

        next_q_action = get_output(self.network, next_states,train=True).max(dim=1,keepdim=True)[0]
        #loss = target - qvalue
        loss = (rewards + self.discount * next_q_action - q_action)
        #mse
        mse = 0.5 * loss ** 2
        #sum loss
        return torch.sum(mse)
    
    def get_loss_sarsa_function(self,states,rewards,next_states,next_actions,actions):
        actionmask = torch.eq(torch.arange(self.nactions).reshape((1, -1)), actions.reshape((-1, 1)))
        actionmask = actionmask.float()
        q_action = (get_output(self.network, states,train=True) * actionmask).sum(axis=1).reshape((-1, 1))
        #q(s_next,a_next)
        next_actionmask = torch.eq(torch.arange(self.nactions).reshape((1, -1)), next_actions.reshape((-1, 1)))
        next_actionmask = next_actionmask.float()
        next_q_action = (get_output(self.network, next_states,train=True) * next_actionmask).sum(axis=1).reshape((-1, 1))
        #loss = target - qvalue
        loss = (rewards + self.discount * next_q_action - q_action)
        #mse
        mse = 0.5 * loss ** 2
        #sum loss
        return torch.sum(mse)
    
    def build_training_sarsa_function(self,states, actions, next_states, next_actions, rewards):
        loss = self.get_loss_sarsa_function(states,rewards,next_states,next_actions,actions)
        loss.backward()
        self._updates_net.step()
        self._updates_net.zero_grad()
        
    def build_training_function(self,states, actions, next_states, rewards):
        loss = self.get_loss_function(states,rewards,next_states,actions)
        loss.backward()
        self._updates_net.step()
        self._updates_net.zero_grad()
    
    def update(self, states, actions, next_states, rewards):        
        states        = torch.tensor(states,dtype=torch.float32).reshape((len(states), self.size))    #f32       #as f32 matrix
        actions       = torch.tensor(actions,dtype=torch.int32).reshape((-1,1))                         #int32               #as column
        next_states   = torch.tensor(next_states,dtype=torch.float32).reshape((len(next_states), self.size)) #as f32 matrix
        rewards       = torch.tensor(rewards,dtype=torch.float32).reshape((-1,1))                            #as column
        self.qtrain(states, actions, next_states, rewards)
        
    def update_sarsa(self,states,actions,next_states,next_actions,rewards):
        states        = torch.tensor(states,dtype=torch.float32).reshape((len(states), self.size))    #f32       #as f32 matrix
        actions       = torch.tensor(actions,dtype=torch.int32).reshape((-1,1))                         #int32               #as column
        next_states   = torch.tensor(next_states,dtype=torch.float32).reshape((len(next_states), self.size)) #as f32 matrix
        next_actions = torch.tensor(next_actions,dtype=torch.int32).reshape((-1,1))   #int32
        rewards       = torch.tensor(rewards,dtype=torch.float32).reshape((-1,1))                            #as column
        self.qtrain(states, actions, next_states,next_actions, rewards)
        
    def params(self):
        return self.network.state_dict()