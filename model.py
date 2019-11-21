from sklearn.datasets import make_moons
import torch
from torch import nn, optim, utils
import torch.nn.functional as F

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
        
        self.optimizer = optim.Adam(self.paramters(),lr=0.001)

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

    def forward(self,x):
        return self.net(x)

class Dataset(utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Trainer():
    def __init__(datasize):
        self.datasize = datasize
        self.criterion = F.cross_entropy()

    def generateData():
        X, y = make_moons(n_samples=self.datasize,noise=0.2)
        
        #define train 80, val 10 and test 10 set from the data 
        
        # DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
        #   batch_sampler=None, num_workers=0, collate_fn=None,
        #   pin_memory=False, drop_last=False, timeout=0,
        #   worker_init_fn=None)





if test:
    X, y = make_moons(n_samples=5,noise=0.2)

    print(X)
    print(y)
    model = Net([20,'Sigmoid',100])
    print(model.net)
    print(model.forward(torch.from_numpy(X).type(torch.FloatTensor)))