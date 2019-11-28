import numpy as np
import torch
from controller import Controller
import matplotlib.pyplot as plt
class trainer():

    def __init__(self):
        self.num_rollouts = 5000
        self.loss_his = []
        self.acc_his = [] 
        self.reward_his = []

    def trainController(self):
        controller = Controller()
        print("Training controller")
        for num_rollout in (range(self.num_rollouts)):
            loss,acc = controller.generate_rollout() 
            self.loss_his.append(loss.numpy())
            self.acc_his.append(acc)
            self.reward_his.append(controller.reward)

            if (num_rollout+1) % (self.num_rollouts/10) == 0:
                print("{}/{}".format((num_rollout+1),self.num_rollouts))
            if (num_rollout+1) % self.num_rollouts == 0:
                print("The last archtecture picked: \n",controller.actions)    


    def plot(self):
        plt.subplot(3,1,1)
        plt.plot(range(self.num_rollouts),self.loss_his)
        plt.xlabel("number of rollouts")
        plt.ylabel("loss over number of rollouts")
        plt.subplot(3,1,2)
        plt.plot(range(self.num_rollouts),self.acc_his)
        plt.xlabel("number of rollouts")
        plt.ylabel("acc over number of rollouts")
        plt.subplot(3,1,3)
        plt.plot(range(self.num_rollouts),self.reward_his)
        plt.xlabel("number of rollouts")
        plt.ylabel("reward over number of rollouts")
        plt.show()

if __name__ == "__main__":
    test = trainer()
    test.trainController()
    test.plot()