from sklearn.datasets import make_moons
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt

test = False
class Dataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Net(nn.Module):

    def __init__(self, string):
        # Inherit from parent constructor
        super(Net, self).__init__()
        self.layers = []
        self.datasize = 1000
        self.epochs = 500
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
        self.layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*self.layers)
        #print('layers', layers)
        self.optimizer = optim.Adam(self.parameters(),lr=0.001)


    def forward(self,x):
        return self.net(x)
    
    def generateData(self,p_val = 0.6):
        X, y = make_moons(n_samples=self.datasize,noise=0.1)
        val = int(len(X)*(1-p_val))
        set_train = Dataset(X[:val],y[:val])
        set_test = Dataset(X[val:],y[val:])
        self.dl_train = data.DataLoader(set_train, batch_size = 32, shuffle=True)
        self.dl_test = data.DataLoader(set_test, batch_size = 32, shuffle=True)


    def train(self):
        self.generateData()
        self.epoch_loss = []
        self.epoch_acc = []
        old_acc = 0
        threshold = 0.01
        max_iter = 0
        for e in range(self.epochs): 
            # --- Trainig ----- #
            loss_e = 0
            for i,data in enumerate(iter(self.dl_train),0):
            
                X,y = data
                pred = self.net(X)

                self.optimizer.zero_grad()
                loss = F.cross_entropy(pred,y.type(torch.LongTensor))
                loss.backward()
                self.optimizer.step()
                loss_e += loss.data.numpy()
                
            self.epoch_loss.append((loss_e/(i+1)))

            # -- Testing -- #
            acc = 0
            for i,data in enumerate(iter(self.dl_test),0):    
                X_test,y_test = data
                pred = self.net(X_test)
                pred = torch.argmax(pred,dim=-1)
                acc += torch.mean(torch.eq(pred,y_test.type(torch.LongTensor)).type(torch.FloatTensor)).data.numpy()
            
            acc /= (i+1)
            self.epoch_acc.append(acc) 

            # -- Early stopping -- #

            if abs(acc-old_acc) < threshold:
                if max_iter == 10:
                    self.epochs = e+1
                    break
                max_iter +=1
            else:
                max_iter = 0
                old_acc = acc 
        
        # --- return acc after trainig --- #

        return acc

    def plot(self):
        print(self.net)
        plt.figure()
        plt.title("loss over epochs")
        plt.plot(range(self.epochs),self.epoch_loss)
        plt.figure()
        plt.title("acc of the archetecture over epochs")
        plt.plot(range(self.epochs),self.epoch_acc)
       


        #define train 80, val 20 and test 0 set from the data 
        
        # DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
        #   batch_sampler=None, num_workers=0, collate_fn=None,
        #   pin_memory=False, drop_last=False, timeout=0,
        #   worker_init_fn=None)





if test:

    net1 = Net([10,'ReLU',10, 'ReLU'])
    net2 = Net([10])
    net3 = Net([])
    avrg = 10
    acc1,acc2,acc3 = 0,0,0
    print("Fitting the model..")
    for _ in range(avrg):
        acc1 += net1.train()  
    print("accuracy from 1",(acc1/avrg))
    for _ in range(avrg):
        acc2 += net2.train()  
    print("accuracy from 2",(acc2/avrg))
    for _ in range(avrg):
        acc3 += net3.train()  
    print("accuracy from 1",(acc3/avrg))
    
   