#!/usr/bin/env python
# coding: utf-8



import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import argparse
import random
import os
import time 
from termcolor import colored
from pickle import dump

import sys
from networks import *

import matplotlib
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
    parser.add_argument('-ntrain', dest='ntrain', type=int, default=2000, help='Number of input field samples used for training (out of available data of the fields). ')
    parser.add_argument('-neval', dest='neval', type=int, default=10, help='Number of random points at which output field is evaluated on each timestamp for a given input field sample during training.')
    parser.add_argument('-seed', dest='seed', type=int, default=0, help='Seed number.')
    parser.add_argument('-use_fourier_features', dest='use_fourier_features', type=bool, default=False, help='Add fourier features at the input of trunk net.')
    args = parser.parse_args()

    # Print all the arguments
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    ntrain = args.ntrain
    neval = args.neval
    seed = args.seed
    use_fourier_features = args.use_fourier_features

    resultdir = os.path.join(os.getcwd(), 'analysis_results', 'ntrain='+str(ntrain)+'-neval='+str(neval)+'-seed='+str(seed)) 
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    orig_stdout = sys.stdout
    q = open(os.path.join(resultdir, 'output-'+'ntrain='+str(ntrain)+'-neval='+str(neval)+'-seed='+str(seed)+'.txt'), 'w')
    sys.stdout = q
    print ("------START------")

    print('ntrain = '+str(ntrain)+', neval = '+str(neval)+', seed = '+str(seed))
    print('use_fourier_features = '+str(use_fourier_features))
    
if save == False:
    ntrain = 2000 # Number of input field samples used for training (out of available data of the fields). 
    neval = 10 # Number of random points at which output field is evaluated on each timestamp for a given input field sample during training.
    seed = 0 # Seed number.
    use_fourier_features = False




start = time.time()
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




# Load the data
path = 'data/'
data = np.load(path+'Diffusion-reaction_dynamics_t=0to1/Diffusion-reaction_dynamics.npz') # Load the .npz file
print(data)
print(data['t_span'].shape)
print(data['x_span'].shape)
print(data['input_s_samples'].shape) # Random Source fields: Gaussian random fields, Nsamples x 100, each sample is (1 x 100)
print(data['output_u_samples'].shape) # Time evolution of the solution field: Nsamples x 101 x 100.
                               # Each field is 101 x 100, rows correspond to time and columns respond to location.
                               # First row corresponds to solution at t=0 (1st time step)
                               # and next  row corresponds to solution at t=0.01 (2nd time step) and so on.
                               # last row correspond to solution at t=1 (101th time step).




# Convert NumPy arrays to PyTorch tensors
inputs = torch.from_numpy(data['input_s_samples']).float().to(device)
outputs = torch.from_numpy(data['output_u_samples']).float().to(device)

t_span = torch.from_numpy(data['t_span']).float().to(device)
x_span = torch.from_numpy(data['x_span']).float().to(device)
nt, nx = len(t_span), len(x_span) # number of discretizations in time and location.
print("nt =",nt, ", nx =",nx)
print("Shape of t-span and x-span:",t_span.shape, x_span.shape)
print("t-span:", t_span)
print("x-span:", x_span)

# Estimating grid points
T, X = torch.meshgrid(t_span, x_span)
# print(T)
# print(X)
grid = torch.vstack((T.flatten(), X.flatten())).T
print("Shape of grid:", grid.shape) # (nt*nx, 2)
print("grid:", grid) # time, location

# Split the data into training and testing sets
inputs_available, inputs_test, outputs_available, outputs_test = train_test_split(inputs, outputs, test_size=500, random_state=seed)

# Check the shapes of the subsets
print("Shape of inputs_available:", inputs_available.shape)
print("Shape of inputs_test:", inputs_test.shape)
print("Shape of outputs_available:", outputs_available.shape)
print("Shape of outputs_test:", outputs_test.shape)
print('#'*100)



class DeepONet(nn.Module):

    def __init__(self, branch_net, trunk_net):
        super().__init__()
        
        self.branch_net = branch_net
        self.trunk_net = trunk_net
    
    def forward(self, branch_inputs, trunk_inputs, test_mode=False):
        """
        bs            :  Batch size.
        m             :  Number of sensors on each input field.
        nt            :  Number of timesteps.
        neval         :  Number of points at which output field is evaluated at each timestep for a given input field sample.
        p             :  Number of output neurons in both branch and trunk net.  
        n_fourier     :  Number of fourier frequencies considered.
        
        branch_inputs shape: (bs, m) 
        trunk_inputs shape : (bs, nt*neval, 2+(4*n_fourier)) # 2 corresponds to t, x 
        
        shapes:  inputs shape                                      -->      outputs shape
        branch:  (bs, m)                                           -->      (bs, p)
        trunk:   (bs, nt*neval, 2+(4*n_fourier))                   -->      (bs, nt*neval, p)
        
        outputs shape: (bs, nt*neval).
        """
        
        branch_outputs = self.branch_net(branch_inputs)
        if test_mode==False:
            trunk_outputs = self.trunk_net(trunk_inputs)
        elif test_mode==True:
            # trunk_inputs here is (nt*neval, 2+(4*n_fourier))
            trunk_outputs = self.trunk_net(trunk_inputs).repeat(branch_inputs.shape[0], 1, 1)
        
        results = torch.einsum('ik, ilk -> il', branch_outputs, trunk_outputs)
        
        return results




class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)




p = 128 # Number of output neurons in both the branch and trunk net.

input_neurons_branch = nx # m
branch_net = DenseNet(layersizes=[input_neurons_branch] + [64]*3 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
branch_net.to(device)
# print(branch_net)
print('BRANCH-NET SUMMARY:')
summary(branch_net, input_size=(input_neurons_branch,))  
print('#'*100)

n_fourier = 10 # Number of fourier frequencies considered.
# 2 corresponds to t, x
if use_fourier_features == False:
    input_neurons_trunk = 2 # Number of input neurons in the trunk net.
else:
    input_neurons_trunk = 2 + (4*n_fourier) # Number of input neurons in the trunk net.
trunk_net = DenseNet(layersizes=[input_neurons_trunk] + [64]*3 + [p], activation=nn.ReLU()) # nn.SiLU()
trunk_net.to(device)
# print(trunk_net)
print('TRUNK-NET SUMMARY:')
summary(trunk_net, input_size=(input_neurons_trunk,))
print('#'*100)

model = DeepONet(branch_net, trunk_net)
model.to(device)




def fourier_features(t, x, n_fourier):
    pi = math.pi
    result = torch.zeros(t.size(0), 2+(4*n_fourier)).to(device)  # Initialize result tensor
    
    # Compute the transformation
    result[:, 0] = t
    result[:, 1] = x
    for i in range(n_fourier):
        result[:, 4*i + 0 + 2] = torch.cos((i + 1) * pi * t)
        result[:, 4*i + 1 + 2] = torch.sin((i + 1) * pi * t)
        result[:, 4*i + 2 + 2] = torch.cos((i + 1) * pi * x)
        result[:, 4*i + 3 + 2] = torch.sin((i + 1) * pi * x)
    return result




print('Select training data from available data of the fields')
random_indices = torch.randperm(inputs_available.shape[0])[:ntrain].to(device) # Generate random row indices
# Select rows from both matrices using the random indices
inputs_train = inputs_available[random_indices]
outputs_train = outputs_available[random_indices]
print('Shape of train data')
print(inputs_train.shape, outputs_train.shape)
print('#'*100)

bs = 256 # Batch size
# Calculate the number of batches
num_batches = len(inputs_train) // bs
# print("Number of batches:", num_batches)
        
# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75000, gamma=1.0)

branch_inputs_test = inputs_test # (bs, m)
# neval = nx
all_evaluation_points = []
for j in range(nt):
    evaluation_points = torch.hstack(( torch.full((len(x_span), 1),t_span[j].item()).to(device), x_span.reshape(-1,1) ))
    all_evaluation_points.append(evaluation_points)
trunk_inputs_test_ = torch.cat(all_evaluation_points, dim=0)
if use_fourier_features == False:
    trunk_inputs_test = trunk_inputs_test_ # (nt*neval, 2) = (nt*nx, 2)
else:
    trunk_inputs_test = fourier_features(trunk_inputs_test_[:, 0], trunk_inputs_test_[:, 1], n_fourier)# (nt*neval, 2+(4*n_fourier)) = (nt*nx, 2+(4*n_fourier))
        

iteration_list, train_loss_list, test_loss_list, learningrates_list = [], [], [], []
iteration = 0

n_epochs = 1000 #10000 #10 # 2000
for epoch in range(n_epochs):
    
    # Shuffle the train data using the generated indices
    num_samples = len(inputs_train)
    indices = torch.randperm(num_samples).to(device) # Generate random permutation of indices
    inputs_train_shuffled = inputs_train[indices]
    outputs_train_shuffled = outputs_train[indices]
    
    # Initialize lists to store batches
    inputs_train_batches = []
    outputs_train_batches = []
    # Split the data into batches
    for i in range(num_batches):
        start_idx = i * bs
        end_idx = (i + 1) * bs
        inputs_train_batches.append(inputs_train_shuffled[start_idx:end_idx])
        outputs_train_batches.append(outputs_train_shuffled[start_idx:end_idx])
    # Handle leftover data into the last batch
    if len(inputs_train_shuffled) % bs != 0:
        start_idx = num_batches * bs
        inputs_train_batches.append(inputs_train_shuffled[start_idx:])
        outputs_train_batches.append(outputs_train_shuffled[start_idx:])
    
    for i, (inputs_batch, outputs_batch) in enumerate(zip(inputs_train_batches, outputs_train_batches)):
        #print(f"Shape of inputs_train_batch[{i}]:", inputs_batch.shape) # (bs, nx)
        #print(f"Shape of outputs_train_batch[{i}]:", outputs_batch.shape) # (bs, nt, nx)
        
        branch_inputs = inputs_batch # (bs, nx)

        all_selected_ids = torch.zeros((inputs_batch.shape[0], nt*neval, 2), dtype=torch.int).to(device) # (bs, nt*neval, 2) # 2 corresponds to time and locations
        if use_fourier_features == False:
            trunk_inputs = torch.zeros((inputs_batch.shape[0], nt*neval, 2), dtype=torch.float).to(device) # (bs, nt*neval, 2)
        else:
            trunk_inputs = torch.zeros((inputs_batch.shape[0], nt*neval, 2 + (4*n_fourier)), dtype=torch.float).to(device) # (bs, nt*neval, 2+(4*n_fourier))
        for j in range(inputs_batch.shape[0]):

            selected_ids_list = []
            for z in range(nt): 
                # 'neval' locations selection for each timestamp
                ids_locations = list(range(len(x_span)))
                selected_ids_locations = torch.tensor(random.sample(ids_locations, neval)).to(device) # random sampling without replacement
                selected_ids = torch.hstack(( torch.full((neval, 1),z).to(device), selected_ids_locations.reshape(-1,1) )) # (neval, 2)
                selected_ids_list.append(selected_ids)

            all_selected_ids[j] = torch.cat(selected_ids_list, dim=0) # (nt*neval, 2)

            all_selected_t_stamps = t_span[all_selected_ids[j][:, 0]]
            all_selected_x_locations = x_span[all_selected_ids[j][:, 1]]
            
            if use_fourier_features == False:
                trunk_inputs[j] = torch.stack((all_selected_t_stamps, all_selected_x_locations), dim=1) # (nt*neval, 3)  
            else:
                trunk_inputs[j] = fourier_features(all_selected_t_stamps, all_selected_x_locations, n_fourier) # (nt*neval, 2+(4*n_fourier))

        outputs_needed = torch.zeros((inputs_batch.shape[0], nt*neval)).to(device) # (bs, nt*neval)
        for k in range(inputs_batch.shape[0]):
            # outputs_batch is (bs, nt, nx)
            outputs_needed[k] = outputs_batch[k, all_selected_ids[k][:, 0], all_selected_ids[k][:, 1]]

        # print(branch_inputs.shape, trunk_inputs.shape, outputs_needed.shape)   
        # print('*********')

        optimizer.zero_grad()
        predicted_values = model(branch_inputs, trunk_inputs) # (bs, nt*neval)
        target_values = outputs_needed # (bs, nt*neval)
        train_loss = nn.MSELoss()(predicted_values, target_values)
        
        predicted_values_test = model(branch_inputs_test, trunk_inputs_test, test_mode=True).reshape(-1, nt, nx)
        target_values_test = outputs_test
        test_loss = nn.MSELoss()(predicted_values_test, target_values_test)
        
        train_loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 200 == 0:
            print('Epoch %s:' % epoch, 'Batch %s:' % i, 'train loss = %f,' % train_loss,
                  'learning rate = %f,' % optimizer.state_dict()['param_groups'][0]['lr'],
                  'test loss = %f' % test_loss) 
        
        iteration_list.append(iteration)
        train_loss_list.append(train_loss.item())
        test_loss_list.append(test_loss.item())
        learningrates_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        iteration+=1
    
if save == True:
    np.save(os.path.join(resultdir,'iteration_list.npy'), np.asarray(iteration_list))
    np.save(os.path.join(resultdir,'train_loss_list.npy'), np.asarray(train_loss_list))
    np.save(os.path.join(resultdir,'test_loss_list.npy'), np.asarray(test_loss_list))
    np.save(os.path.join(resultdir,'learningrates_list.npy'), np.asarray(learningrates_list))
    
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

plt.figure()
plt.plot(iteration_list, learningrates_list, 'b', label = 'learning-rate')
plt.xlabel('Iterations')
plt.ylabel('Learning-rate')
plt.legend()
plt.tight_layout()
if save == True:
    plt.savefig(os.path.join(resultdir,'learning-rate_plot.pdf'))
    
# end timer
finish = time.time() - start  # time for network to train




if save == True:
    torch.save(model.state_dict(), os.path.join(resultdir,'model_state_dict.pt'))
# model.load_state_dict(torch.load(os.path.join(resultdir,'model_state_dict.pt')))




# Predictions to save
predicted_values_test = model(branch_inputs_test, trunk_inputs_test, test_mode=True).reshape(-1, nt, nx).cpu()
target_values_test = outputs_test.cpu()
test_loss = nn.MSELoss()(predicted_values_test, target_values_test)
print("Mean Squared Error Test:\n", test_loss.item())

if save == True:
    np.save(os.path.join(resultdir, 'inputs_test.npy'), inputs_test.cpu().detach().numpy())
    np.save(os.path.join(resultdir, 'predicted_values_test.npy'), predicted_values_test.detach().numpy())
    np.save(os.path.join(resultdir, 'target_values_test.npy'), target_values_test.detach().numpy())




# Predictions
r2score_list = []
mse_list = []

for i in range(inputs_test.shape[0]):
    
    branch_inputs = inputs_test[i].reshape(1, nx) # (bs, m) = (1, nx) 
    trunk_inputs = grid.unsqueeze(0) # (bs, nt*neval, 2) = (1, nt*nx, 2)

    prediction_i = model(branch_inputs, trunk_inputs) # (bs, nt*neval) = (1, nt*nx)
    target_i = outputs_test[i].reshape(1, -1) # (1, nt*nx)
    
    r2score_i = metrics.r2_score(target_i.flatten().cpu().detach().numpy(), prediction_i.flatten().cpu().detach().numpy()) 
    r2score_list.append(r2score_i)
    
    mse_i = F.mse_loss(prediction_i.cpu(), target_i.cpu())
    mse_list.append(mse_i.item())
    
    if (i+1) % 20 == 0:
        print(colored('TEST SAMPLE '+str(i+1), 'red'))
    
        r2score = metrics.r2_score(target_i.flatten().cpu().detach().numpy(), prediction_i.flatten().cpu().detach().numpy()) 
        relerror = np.linalg.norm(target_i.flatten().cpu().detach().numpy() - prediction_i.flatten().cpu().detach().numpy()) / np.linalg.norm(target_i.flatten().cpu().detach().numpy())
        r2score = float('%.4f'%r2score)
        relerror = float('%.4f'%relerror)
        print('Rel. L2 Error = '+str(relerror)+', R2 score = '+str(r2score))

        fig = plt.figure(figsize=(15,3.5))
        plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.5, wspace = 0.4, hspace = 0.1)

        ax = fig.add_subplot(1, 4, 1)    
        ax.scatter(x_span.cpu().detach().numpy(), inputs_test[i].cpu().detach().numpy(), color='k', s=5)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$s(x)$', rotation="horizontal")
        ax.yaxis.set_label_coords(-0.20, 0.5)
        ax.set_title('Source field')

        ax = fig.add_subplot(1, 4, 2)  
        cnt = plt.contourf(X.cpu().detach().numpy(), T.cpu().detach().numpy(), target_i.reshape(nt, nx).cpu().detach().numpy(), levels=100, cmap='seismic')
        cnt.set_edgecolor("face")
        cbar = plt.colorbar(cnt, format='%.2f')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$t$', rotation="horizontal")
        ax.yaxis.set_label_coords(-0.15, 0.5)
        ax.set_title('True field')

        ax = fig.add_subplot(1, 4, 3)  
        cnt = plt.contourf(X.cpu().detach().numpy(), T.cpu().detach().numpy(), prediction_i.reshape(nt, nx).cpu().detach().numpy(), levels=100, cmap='seismic')
        cnt.set_edgecolor("face")
        cbar = plt.colorbar(cnt, format='%.2f')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$t$', rotation="horizontal")
        ax.yaxis.set_label_coords(-0.15, 0.5)
        ax.set_title('Predicted field') 

        ax = fig.add_subplot(1, 4, 4)  
        cnt = plt.contourf(X.cpu().detach().numpy(), T.cpu().detach().numpy(), np.abs(target_i.reshape(nt, nx).cpu().detach().numpy() - prediction_i.reshape(nt, nx).cpu().detach().numpy()), levels=100, cmap='seismic')
        cnt.set_edgecolor("face")
        plt.colorbar(cnt, format='%.4f')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$t$', rotation="horizontal")
        ax.yaxis.set_label_coords(-0.15, 0.5)
        ax.set_title('Absolute error')

        #sns.despine(trim=True)
        plt.tight_layout()

        if save == True:
            plt.savefig(os.path.join(resultdir,'Test_Sample_'+str(i+1)+'.pdf'))
            plt.show()
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





