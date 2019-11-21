from sklearn.datasets import make_moons
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
from torch.utils import data

test = True

class Net(nn.Module):

    def __init__(self, string):
        # Inherit from parent constructor
        super(Net, self).__init__()
        self.layers = []
        self.in_features = 2
        self.out_classes = 2
        self.act_dict = {
            'ReLU':nn.ReLU(),
            'Tanh':nn.Tanh(),
            'Sigmoid':nn.Sigmoid(),
        }
        

        num_input = self.in_features
        
        # Break down string sent from Controller
        # and add layers in network based on this
        for s in string:
            # If element in string is not a number (i.e. an activation function)
            try:
                s_int = int(s)
                self.layers.append(nn.Linear(num_input, s_int))
                num_input = s_int
            except:
                self.layers.append(self.act_dict[s])
                
        # Last layer with output 2 representing the two classes
        self.layers.append(nn.Linear(num_input, self.out_classes))
        self.layers.append(nn.Softmax(dim=0))
        self.net = nn.Sequential(*self.layers)
        #print('layers', layers)
        self.optimizer = optim.Adam(self.parameters(),lr=0.001)


    def forward(self,x):
        return self.net(x)

class Dataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Trainer():
    def __init__(self,datasize,arch_str):
        self.datasize = datasize
        self.arch_str = arch_str
        self.net = Net(arch_str)
    def generateData(self,p_val = 0.2):
        X, y = make_moons(n_samples=self.datasize,noise=0.2)
        val = int(len(X)*(1-p_val))
        set_train = Dataset(X[:val],y[:val])
        set_test = Dataset(X[val:],y[val:])
        self.dl_train = data.DataLoader(set_train, batch_size = 32, shuffle=True)
        self.dl_test = data.DataLoader(set_test, batch_size = 32, shuffle=True)


    def train(self):
        self.generateData()
        epcohs = 50

        loss_train = []

        # --- Trainig ----- #
        for i,data in enumerate(iter(self.dl_train),0):
            X,y = data
            pred = self.net.forward(self.arch_str)
            self.net.optimizer.zero_grad()
            loss = F.cross_entropy(pred,y.type(torch.LongTensor))
            loss.backward()
            self.net.optimizer.step()
            print(loss.data)



        #define train 80, val 20 and test 0 set from the data 
        
        # DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
        #   batch_sampler=None, num_workers=0, collate_fn=None,
        #   pin_memory=False, drop_last=False, timeout=0,
        #   worker_init_fn=None)





if test:
    train = Trainer(500,[])
    train.train()