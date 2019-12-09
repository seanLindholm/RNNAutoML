import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from model import Net
from torch.autograd import Variable
import numpy as np

test = False
class Controller(nn.Module):

    def __init__(self):
        # Inherit from parent constructor
        super(Controller, self).__init__()
        self.nodeSize = {
            0: "term",
            1: 2,
            2: 4,
            3: 8,
            4: 16,
            5: 32,
            6: 64
        }
        self.activations = {
            0: "ReLU",
            1: "Tanh",
            2: "Sigmoid",
        }
        self.hidden_dim = 50
        self.max_depth = 12
        self.acc_tests = 5
        self.GRU = nn.GRUCell(input_size=self.hidden_dim,hidden_size = self.hidden_dim)
        
        self.decoder = []
        for i in range(self.max_depth):
            if i%2 == 0:
                self.decoder.append(nn.Linear(self.hidden_dim,len(self.nodeSize)))
            else:
                self.decoder.append(nn.Linear(self.hidden_dim,len(self.activations)))

        
        self.optimizer = optim.Adam(self.parameters(),lr=0.001)
        self.epsilon = 1

    def forward(self,x,h,depth):
        x = x.unsqueeze(0)
        h = h.unsqueeze(0)
        h = self.GRU(x,h)
        x = self.decoder[depth](h)
        

        return x.squeeze(),h.squeeze() 


    def step(self,state,depth):
        logits,new_state = self.forward(torch.zeros(self.hidden_dim),state,depth)
        self.probs = F.softmax(logits,dim=-1)
        log_probs = F.log_softmax(logits,dim=-1)
        choice = self.probs.multinomial(num_samples=1).data
        if depth %2 == 0:
            action = self.nodeSize[int(choice)]
        else:
            action = self.activations[int(choice)]
        act_log_prob = log_probs[int(choice)]
        return action, act_log_prob ,new_state

    def generate_rollout(self):
        # -- init state -- #
        self.states = []
        self.actions = []
        self.log_probs = []
        self.state_entropy = []
        self.reward=0
        state = torch.zeros(self.hidden_dim)
        self.states.append(state)
        depth = 0
        while True and self.max_depth > depth:
            action,log_prob,next_state = self.step(state,depth)
            state = next_state
          
            if action == "term":
                self.log_probs.append(log_prob)
                self.states.append(state)
                self.state_entropy.append(torch.mul(log_prob.sum(),self.probs.sum()))            
                break

            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.states.append(state)

            self.state_entropy.append(-torch.mul(log_prob.sum(),self.probs.sum()))
            depth+=1

        net = Net(self.actions) 
        acc = 0
        for _ in range(self.acc_tests):
            acc += net.train()
        acc /= self.acc_tests
        self.reward += acc
        return self.optimize(),acc
       


    def optimize(self):
        # self.log_probs = torch.cat(self.log_probs)
        # self.state_entropy = torch.cat(self.state_entropy)
        # loss = -torch.mean(torch.mul(self.log_probs,(R + (self.epsilon * self.state_entropy) )))
        
        loss = 0
        R = torch.ones(1)*self.reward 
      
        for i in range(len(self.log_probs)):
             loss -= self.log_probs[i]*Variable(R) - self.epsilon*self.state_entropy[i]
             

        loss /= len(self.log_probs)
        self.optimizer.zero_grad()
        loss.backward()    
        self.optimizer.step()
        self.epsilon *= 0.99
        return loss.data
        
       
        


            




if test:
    controller = Controller()
    controller.generate_rollout()
    
    