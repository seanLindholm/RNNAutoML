import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from model import Net
from torch.autograd import Variable

test = False
class Controller(nn.Module):

    def __init__(self):
        # Inherit from parent constructor
        super(Controller, self).__init__()
        self.action_space = {
            0: "term",
            1: 2,
            2: 4,
            3: 8,
            4: "ReLU",
            5: "Sigmoid",
            6: "Tanh",
        }
        self.hidden_dim = 50
        self.max_depth = 12
        self.acc_tests = 5
        self.GRU = nn.GRUCell(input_size=len(self.action_space),hidden_size = self.hidden_dim)
        self.fcl = nn.Linear(self.hidden_dim,len(self.action_space))
        self.optimizer = optim.Adam(self.parameters(),lr=0.001)


    def forward(self,x,h):
        x = x.unsqueeze(0)
        h = h.unsqueeze(0)
        
        h = self.GRU(x,h)
        x = self.fcl(h)
        

        return x.squeeze(),h.squeeze() 


    def step(self,state):
        logits,new_state = self.forward(torch.zeros(len(self.action_space)),state)
        probs = F.softmax(logits,dim=-1)
        log_probs = F.log_softmax(logits,dim=-1)
        choice = probs.multinomial(num_samples=1).data
        action = self.action_space[int(choice)]
        act_log_prob = log_probs[choice]
    
        
        return action, act_log_prob ,new_state

    def generate_rollout(self):
        # -- init state -- #
        self.states = []
        self.actions = []
        self.log_probs = []
        self.reward=0
        state = torch.zeros(self.hidden_dim)
        self.states.append(state)
        depth = 0
        while True and self.max_depth > depth:
            action,log_prob,next_state = self.step(state)
            state = next_state
            if action == "term":
                self.log_probs.append(log_prob)
                self.states.append(state)
                if len(self.actions) == 0:
                    self.reward -= 1
                break               
            
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.states.append(state)
            depth+=1

            if len(self.actions) >= 2 and isinstance(self.actions[-2],str) and isinstance(self.actions[-1],str):
                self.reward -=0.2
            
            elif len(self.actions) >= 2 and isinstance(self.actions[-2],int) and isinstance(self.actions[-1],int):
                self.reward -=0.2

        net = Net(self.actions) 
        acc = 0
        for _ in range(self.acc_tests):
            acc += net.train()
        acc /= self.acc_tests
        self.reward += acc
        return self.optimize(),acc
       


    def optimize(self):
        #self.log_probs = torch.cat(self.log_probs)   
        R = torch.ones(1)*self.reward 
        loss = 0
        for i in range(len(self.log_probs)):
            loss -= self.log_probs[i]*Variable(R)

        loss /= len(self.log_probs)
        self.optimizer.zero_grad()
        loss.backward()    
        self.optimizer.step()
        return loss.data
        
       
        


            




if test:
    controller = Controller()
    controller.generate_rollout()
    
    