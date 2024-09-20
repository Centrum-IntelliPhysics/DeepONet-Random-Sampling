#!/usr/bin/env python
# coding: utf-8



import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import metrics
import argparse
import random
import os
import time 
from termcolor import colored

import sys
from networks import *

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9})
import seaborn as sns
sns.set_style("white")
sns.set_style("ticks")

import warnings
warnings.filterwarnings("ignore")




save = True




if save == True:
    parser = argparse.ArgumentParser()
    parser.add_argument('-ntrain', dest='ntrain', type=int, default=8000, help='Number of input field samples used for training (out of available data of the fields).')
    parser.add_argument('-neval', dest='neval', type=int, default=1, help='Number of random points at which output field is evaluated for a given input field sample during training.')
    parser.add_argument('-seed', dest='seed', type=int, default=0, help='Seed number.')
    args = parser.parse_args()

    # Print all the arguments
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    ntrain = args.ntrain
    neval = args.neval
    seed = args.seed

    resultdir = os.path.join(os.getcwd(), 'analysis_results', 'ntrain='+str(ntrain)+'-neval='+str(neval)+'-seed='+str(seed)) 
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    orig_stdout = sys.stdout
    q = open(os.path.join(resultdir, 'output-'+'ntrain='+str(ntrain)+'-neval='+str(neval)+'-seed='+str(seed)+'.txt'), 'w')
    sys.stdout = q
    print ("------START------")

    print('ntrain = '+str(ntrain)+', neval = '+str(neval)+', seed = '+str(seed))
    
if save == False:
    ntrain = 8000 # Number of input field samples used for training (out of available data of the fields). 
    neval = 1 # Number of random points at which output field is evaluated for a given input field sample during training.
    seed = 0 # Seed number.




start = time.time()
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




class DeepONet(nn.Module):

    def __init__(self, branch_net, trunk_net):
        super().__init__()
        
        self.branch_net = branch_net
        self.trunk_net = trunk_net
    
    def forward(self, branch_inputs, trunk_inputs, test_mode=False):
        """
        bs    :  Batch size.
        m     :  Number of sensors on each input field.
        neval :  Number of points at which output field is evaluated for a given input field sample.
        p     :  Number of output neurons in both branch and trunk net.   
        
        branch_inputs shape: (bs x 1 x m) 
        trunk_inputs shape : (bs x neval x 1)
        
        shapes:  inputs shape         -->      outputs shape
        branch:  (bs x 1 x m)         -->      (bs x 1 x p)
        trunk:   (bs x neval x 1)     -->      (bs x neval x p)
        
        outputs shape: (bs x neval).
        """
        
        branch_outputs = self.branch_net(branch_inputs)
        if test_mode==False:
            trunk_outputs = self.trunk_net(trunk_inputs)
        elif test_mode==True:
            # trunk_inputs here is (neval, 1)
            trunk_outputs = self.trunk_net(trunk_inputs).repeat(branch_inputs.shape[0], 1, 1)
        
        results = torch.einsum('ijk, ilk -> il', branch_outputs, trunk_outputs)
        
        return results




"""
input_neurons_branch: Number of input neurons in the branch net.
input_neurons_trunk: Number of input neurons in the trunk net.
output_neurons: Number of output neurons in both the branch and trunk net.
"""
input_neurons_branch = 100 # m
input_neurons_trunk = 1
output_neurons = 40 # p

branch_layersizes = [input_neurons_branch] + [40]*3 + [output_neurons]  
branch_net = DenseNet(layersizes=branch_layersizes, activation=nn.ReLU()) #nn.SiLU()

trunk_layersizes = [input_neurons_trunk] + [40]*3 + [output_neurons]  
trunk_net = DenseNet(layersizes=trunk_layersizes, activation=nn.ReLU()) #nn.SiLU()

model = DeepONet(branch_net, trunk_net).to(device)




# Load the data
cellcenters_data = np.load('data/cellcenters.npy')
u_data, s_data = np.load('data/u_samples.npy'), np.load('data/s_samples.npy')

# Convert NumPy arrays to PyTorch tensors
cellcenters_tensor = torch.from_numpy(cellcenters_data).float().to(device)
u_tensor = torch.from_numpy(u_data).float().to(device)
s_tensor = torch.from_numpy(s_data).float().to(device)

# Split the data into training (8000) and testing (2000) sets
u_available, u_test, s_available, s_test = train_test_split(u_tensor, s_tensor, test_size=2000, random_state=seed)

# Check the shapes of the subsets
print("Shape of cellcenters:", cellcenters_tensor.shape)
print("Shape of u_available:", u_available.shape)
print("Shape of u_test:", u_test.shape)
print("Shape of s_available:", s_available.shape)
print("Shape of s_test:", s_test.shape)




# Select training data from available data of the fields
random_indices = torch.randperm(u_available.shape[0])[:ntrain].to(device) # Generate random row indices
# Select rows from both matrices using the random indices
u_train = u_available[random_indices]
s_train = s_available[random_indices]
# print(u_train.shape, s_train.shape)

bs = 256 # Batch size
# Calculate the number of batches
num_batches = len(u_train) // bs
# print("Number of batches:", num_batches)
        
# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

branch_inputs_test = u_test.unsqueeze(1) # (bs, 1, m)
trunk_inputs_test = cellcenters_tensor.reshape(-1, 1) # (neval, 1) = (100, 1)

iteration_list, train_loss_list, test_loss_list = [], [], []
iteration = 0

n_epochs = 1000
for epoch in range(n_epochs):
    
    # Shuffle the train data using the generated indices
    num_samples = len(u_train)
    indices = torch.randperm(num_samples).to(device) # Generate random permutation of indices
    u_train_shuffled = u_train[indices]
    s_train_shuffled = s_train[indices]
    
    # Initialize lists to store batches
    u_train_batches = []
    s_train_batches = []
    # Split the data into batches
    for i in range(num_batches):
        start_idx = i * bs
        end_idx = (i + 1) * bs
        u_batch = u_train_shuffled[start_idx:end_idx]
        s_batch = s_train_shuffled[start_idx:end_idx]
        u_train_batches.append(u_batch)
        s_train_batches.append(s_batch)
    # Handle leftover data into the last batch
    if len(u_train_shuffled) % bs != 0:
        start_idx = num_batches * bs
        u_train_batches.append(u_train_shuffled[start_idx:])
        s_train_batches.append(s_train_shuffled[start_idx:])
    
    for i, (u_batch, s_batch) in enumerate(zip(u_train_batches, s_train_batches)):
        #print(f"Shape of u_train_batch[{i}]:", u_batch.shape) # (bs, m)
        #print(f"Shape of s_train_batch[{i}]:", s_batch.shape) # (bs, m)
        
        branch_inputs = u_batch.unsqueeze(1) # (bs, 1, m)
        
        selected_integers = torch.zeros((u_batch.shape[0], neval), dtype=torch.int) # (bs, neval)
        for j in range(u_batch.shape[0]):
            integers = list(range(len(cellcenters_tensor)))
            sampled_integers = random.sample(integers, neval) # random sampling without replacement.
            selected_integers[j] = torch.tensor(sampled_integers)
        trunk_inputs = cellcenters_tensor[selected_integers].unsqueeze(2) # (bs, neval, 1)

        s_needed = torch.zeros((u_batch.shape[0], neval)).to(device) # (bs, neval)
        for k in range(u_batch.shape[0]):
            s_needed[k] = s_batch[k, selected_integers[k]]

        #print(branch_inputs.shape, trunk_inputs.shape, s_needed.shape)   
        #print('*********')
        
        optimizer.zero_grad()
        predicted_values = model(branch_inputs, trunk_inputs) # (bs, neval)
        target_values = s_needed # (bs, neval)
        train_loss = nn.MSELoss()(predicted_values, target_values)
        
        predicted_values_test = model(branch_inputs_test, trunk_inputs_test, test_mode=True)
        target_values_test = s_test
        test_loss = nn.MSELoss()(predicted_values_test, target_values_test)
        
        train_loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
        
        if epoch % 200 == 0:
            print('Epoch %s:' % epoch, 'Batch %s:' % i, 'train loss = %f,' % train_loss, 'test loss = %f' % test_loss) 
        
        iteration_list.append(iteration)
        train_loss_list.append(train_loss.item())
        test_loss_list.append(test_loss.item())
        iteration+=1

if save == True:
    np.save(os.path.join(resultdir,'iteration_list.npy'), np.asarray(iteration_list))
    np.save(os.path.join(resultdir,'train_loss_list.npy'), np.asarray(train_loss_list))
    np.save(os.path.join(resultdir,'test_loss_list.npy'), np.asarray(test_loss_list))
    
plt.figure()
plt.plot(iteration_list, train_loss_list, 'g', label = 'training loss')
plt.yscale("log")
plt.xlabel('Iterations')
plt.ylabel('Training loss')
plt.legend()
plt.tight_layout()
if save == True:
    plt.savefig(os.path.join(resultdir,'train_loss_plot.pdf'))
    
plt.figure()
plt.plot(iteration_list, test_loss_list, 'r', label = 'testing loss')
plt.yscale("log")
plt.xlabel('Iterations')
plt.ylabel('Testing loss')
plt.legend()
plt.tight_layout()
if save == True:
    plt.savefig(os.path.join(resultdir,'test_loss_plot.pdf'))
    
# end timer
finish = time.time() - start  # time for network to train




if save == True:
    torch.save(model.state_dict(), os.path.join(resultdir,'model_state_dict.pt'))
# model.load_state_dict(torch.load(os.path.join(resultdir,'model_state_dict.pt')))




# Predictions to save
predicted_values_test = model(branch_inputs_test, trunk_inputs_test, test_mode=True).cpu()
target_values_test = s_test.cpu()
test_loss = nn.MSELoss()(predicted_values_test, target_values_test)
print("Mean Squared Error Test:\n", test_loss.item())

if save == True:
    np.save(os.path.join(resultdir, 'u_test.npy'), u_test.cpu().detach().numpy())
    np.save(os.path.join(resultdir, 'predicted_values_test.npy'), predicted_values_test.detach().numpy())
    np.save(os.path.join(resultdir, 'target_values_test.npy'), target_values_test.detach().numpy())




# Predictions
r2score_list = []
mse_list = []

for i in range(u_test.shape[0]):
    
    # bs = 1, m = 100, neval = 100
    branch_inputs = u_test[i].reshape(1, 1, -1) # (bs, 1, m) = (1, 1, 100)
    trunk_inputs= cellcenters_tensor.reshape(1, -1, 1) # (bs, neval, 1) = (1, 100, 1)
    predicted_values = model(branch_inputs, trunk_inputs) # (bs, neval) = (1, 100)
    prediction_i = predicted_values
    target_i = s_test[i].reshape(1, -1)

    r2score_i = metrics.r2_score(target_i.flatten().cpu().detach().numpy(), prediction_i.flatten().cpu().detach().numpy()) 
    r2score_list.append(r2score_i)
    
    mse_i = F.mse_loss(prediction_i.cpu(), target_i.cpu())
    mse_list.append(mse_i.item())

    if (i+1) % 100 == 0:
        print(colored('TEST SAMPLE '+str(i+1), 'red'))
        r2score = metrics.r2_score(target_i.flatten().cpu().detach().numpy(), prediction_i.flatten().cpu().detach().numpy()) 
        relerror = np.linalg.norm(target_i.flatten().cpu().detach().numpy() - prediction_i.flatten().cpu().detach().numpy()) / np.linalg.norm(target_i.flatten().cpu().detach().numpy())
        r2score = float('%.4f'%r2score)
        relerror = float('%.4f'%relerror)
        print('Rel. L2 Error = '+str(relerror)+', R2 score = '+str(r2score))
        
        plt.figure(figsize=(8, 6))
        plt.plot(cellcenters_tensor.cpu().detach().numpy(), u_test[i].cpu().detach().numpy(), '--' ,label='Input', color='green')
        plt.plot(cellcenters_tensor.cpu().detach().numpy(), prediction_i.flatten().cpu().detach().numpy(), label='Prediction', color='red')
        plt.plot(cellcenters_tensor.cpu().detach().numpy(), s_test[i].cpu().detach().numpy(), 'x', markersize=2, label='Truth', color='blue')
        plt.xlabel('x')
        plt.ylabel('Value')
        #plt.title('Test Sample ' + str(i+1))
        plt.legend()
        if save == True:
            plt.savefig(os.path.join(resultdir,'Test_Sample_'+str(i+1)+'.pdf')) 
            plt.close()
        if save == False:
            plt.show()
            
        print(colored('#'*230, 'green'))

mean_r2score = sum(r2score_list) / len(r2score_list)
print("Mean R2 score Test:\n", mean_r2score)
        
mse = sum(mse_list) / len(mse_list)
print("Mean Squared Error Test:\n", mse)




print("Time (sec) to complete:\n" +str(finish)) # time for network to train
if save == True:
    print ("------END------")
    sys.stdout = orig_stdout
    q.close()






