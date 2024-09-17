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
from fipy import *

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
    parser.add_argument('-ntrain', dest='ntrain', type=int, default=5000, help='Number of input field samples used for training (out of available data of the fields).')
    parser.add_argument('-neval', dest='neval', type=int, default=16, help='Number of random points at which output field is evaluated for a given input field sample during training.')
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
    ntrain = 5000 # Number of input field samples used for training (out of available data of the fields).
    neval = 16 # Number of random points at which output field is evaluated for a given input field sample during training.
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

# Convert NumPy arrays to PyTorch tensors
inputs = torch.from_numpy(np.load(path+'conductivity_samples_RBF_nx1=32_nx2=32_lx1=0.1_lx2=0.15_v=1_num_samples=5500.npy')).float().to(device)
outputs = torch.from_numpy(np.load(path+'temperature_samples_RBF_nx1=32_nx2=32_lx1=0.1_lx2=0.15_v=1_num_samples=5500.npy')).float().to(device)
print("Shape of inputs:", inputs.shape)
print("Shape of outputs:", outputs.shape)
print('#'*100)

# Split the data into training and testing sets
inputs_available, inputs_test, outputs_available, outputs_test = train_test_split(inputs, outputs, test_size=500, random_state=seed)

# Check the shapes of the subsets
print("Shape of inputs_available:", inputs_available.shape)
print("Shape of inputs_test:", inputs_test.shape)
print("Shape of outputs_available:", outputs_available.shape)
print("Shape of outputs_test:", outputs_test.shape)
print('#'*100)

# Cellcenters/pixelcenters/gridcenters/meshcenters/mesh_locations
#################################################################
nx1 = int(math.sqrt(inputs_available.shape[1])) # Number of cells in the horizontal direction or Width of image
nx2 = int(math.sqrt(inputs_available.shape[1])) # Number of cells in the vertical direction or Height of image

locations = torch.from_numpy(np.load(path+'cellcenters_nx1=32_nx2=32.npy')).float().to(device) # (nx1*nx2,2) matrix
print("Shape of locations:", locations.shape)
print("locations:", locations)
x1_locations = locations[:,0]
x2_locations = locations[:,1]
# print("x1 locations:", x1_locations)
# print("x2 locations:", x2_locations)
print("Shape of x1 and x2 locations:", x1_locations.shape, x2_locations.shape)
print('#'*100)




class DeepONet(nn.Module):

    def __init__(self, branch_net, trunk_net):
        super().__init__()

        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def forward(self, branch_inputs, trunk_inputs, test_mode=False):
        """
        bs            :  Batch size.
        c             :  Number of channels in input field image.
        h             :  Number of pixels along the height of input field image.
        w             :  Number of pixels along the width of input field image.
        neval         :  Number of points at which output field is evaluated for a given input field sample.
        p             :  Number of output neurons in both branch and trunk net.
        n_fourier     :  Number of fourier frequencies considered.

        branch_inputs shape: (bs, c, h, w)
        trunk_inputs shape : (bs, neval, 2+(4*n_fourier)) # 2 corresponds to x1 and x2

        shapes:  inputs shape                        -->      outputs shape
        branch:  (bs, c, h, w)                       -->      (bs, p)
        trunk:   (bs, neval, 2+(4*n_fourier))        -->      (bs, neval, p)

        outputs shape: (bs, neval).
        """

        branch_outputs = self.branch_net(branch_inputs)
        if test_mode==False:
            trunk_outputs = self.trunk_net(trunk_inputs)
        elif test_mode==True:
            # trunk_inputs here is (neval, 2+(4*n_fourier))
            trunk_outputs = self.trunk_net(trunk_inputs).repeat(branch_inputs.shape[0], 1, 1)

        results = torch.einsum('ik, ilk -> il', branch_outputs, trunk_outputs)

        return results




class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)




p = 150 # Number of output neurons in both the branch and trunk net.

input_size = (nx2, nx1) # Specify input size of image as a tuple (height, width)
n_channels = 1
num_filters = [40, 60, 100]
filter_sizes = [3, 3, 3]
strides = [1]*len(num_filters)
paddings = [0]*len(num_filters)
poolings = [('avg', 2, 2), ('avg', 2, 2), ('avg', 2, 2)]  # Pooling layer specification (type, kernel_size, stride)
end_MLP_layersizes = [150, 150, p]
activation = nn.ReLU() # nn.SiLU()
branch_net = ConvNet(input_size, n_channels, num_filters, filter_sizes, strides, paddings, poolings, end_MLP_layersizes, activation)
branch_net.to(device)
# print(branch_net)
print('BRANCH-NET SUMMARY:')
summary(branch_net, input_size=(n_channels, nx2, nx1))  # input shape is (channels, height, width)
print('#'*100)

n_fourier = 10 # Number of fourier frequencies considered.
# 2 corresponds to x1 and x2 
if use_fourier_features == False:
    input_neurons_trunk = 2 # Number of input neurons in the trunk net.
else:
    input_neurons_trunk = 2 + (4*n_fourier) # Number of input neurons in the trunk net.
trunk_net = DenseResNet(dim_in=input_neurons_trunk, dim_out=p, num_resnet_blocks=2, 
                 num_layers_per_block=2, num_neurons=150, activation=nn.ReLU()) # nn.SiLU()
trunk_net.to(device)
# print(trunk_net)
print('TRUNK-NET SUMMARY:')
summary(trunk_net, input_size=(input_neurons_trunk,))
print('#'*100)

model = DeepONet(branch_net, trunk_net)
model.to(device)




def fourier_features(x1, x2, n_fourier):
    pi = math.pi
    result = torch.zeros(x1.size(0), 2+(4*n_fourier)).to(device)  # Initialize result tensor
    
    # Compute the transformation
    result[:, 0] = x1
    result[:, 1] = x2
    for i in range(n_fourier):
        result[:, 4*i + 0 + 2] = torch.cos((i + 1) * pi * x1)
        result[:, 4*i + 1 + 2] = torch.sin((i + 1) * pi * x1)
        result[:, 4*i + 2 + 2] = torch.cos((i + 1) * pi * x2)
        result[:, 4*i + 3 + 2] = torch.sin((i + 1) * pi * x2)
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75000, gamma=1.0)

branch_inputs_test = inputs_test.reshape(-1, n_channels, nx2, nx1) # (bs, c, h, w) = (testsize, n_channels, nx2, nx1)
if use_fourier_features == False:
    trunk_inputs_test = locations # (neval, 2) = (nx1*nx2, 2)
else:
    trunk_inputs_test = fourier_features(x1_locations, x2_locations, n_fourier) # (neval, 2+(4*n_fourier)) = (nx1*nx2, 2+(4*n_fourier))

iteration_list, train_loss_list, test_loss_list, learningrates_list = [], [], [], []
iteration = 0

n_epochs = 20000 # 2000
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
        #print(f"Shape of inputs_train_batch[{i}]:", inputs_batch.shape) # (bs, nx1*nx2)
        #print(f"Shape of outputs_train_batch[{i}]:", outputs_batch.shape) # (bs, nx1*nx2)

        branch_inputs = inputs_batch.reshape(-1, n_channels, nx2, nx1) # (bs, no. of channels, height, width)

        all_selected_ids = torch.zeros((inputs_batch.shape[0], neval, 1), dtype=torch.int).to(device) # (bs, neval, 1) # 1 corresponds to locations
        if use_fourier_features == False:
            trunk_inputs = torch.zeros((inputs_batch.shape[0], neval, 2), dtype=torch.float).to(device) # (bs, neval, 2)
        else:
            trunk_inputs = torch.zeros((inputs_batch.shape[0], neval, 2 + (4*n_fourier)), dtype=torch.float).to(device) # (bs, neval, 2+(4*n_fourier))
        for j in range(inputs_batch.shape[0]):
            # 'neval' locations selection
            ids_locations = list(range(len(locations)))
            selected_ids_locations = torch.tensor(random.sample(ids_locations, neval)).to(device) # random sampling without replacement
            all_selected_ids[j] = selected_ids_locations.reshape(-1,1) # (neval, 1)
            all_selected_x1_locations = x1_locations[all_selected_ids[j][:, 0]] # (neval)
            all_selected_x2_locations = x2_locations[all_selected_ids[j][:, 0]] # (neval)
            if use_fourier_features == False:
                trunk_inputs[j] = torch.stack((all_selected_x1_locations, all_selected_x2_locations), dim=1) # (neval, 2)
            else:
                trunk_inputs[j] = fourier_features(all_selected_x1_locations, all_selected_x2_locations, n_fourier) # (neval, 2+(4*n_fourier))

        outputs_needed = torch.zeros((inputs_batch.shape[0], neval)).to(device) # (bs, neval)
        for k in range(inputs_batch.shape[0]):
            # outputs_batch is (bs, nx1*nx2) shape
            outputs_needed[k] = outputs_batch[k, all_selected_ids[k][:, 0]]

        # print(branch_inputs.shape, trunk_inputs.shape, outputs_needed.shape)
        # print('*********')

        optimizer.zero_grad()
        predicted_values = model(branch_inputs, trunk_inputs) # (bs, neval)
        target_values = outputs_needed # (bs, neval)
        # L2 regularization (also known as weight decay)
        l2_lambda = 0 # 0.01  # Regularization strength
        l2_reg = torch.tensor(0.0).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)**2
        train_loss = nn.MSELoss()(predicted_values, target_values) + (l2_lambda * l2_reg)
        
        predicted_values_test = model(branch_inputs_test, trunk_inputs_test, test_mode=True)
        target_values_test = outputs_test
        test_loss = nn.MSELoss()(predicted_values_test, target_values_test)
        
        train_loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:
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
predicted_values_test = model(branch_inputs_test, trunk_inputs_test, test_mode=True).cpu()
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

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["navy","blue","deepskyblue","limegreen","yellow","darkorange","red","maroon"])

for i in range(inputs_test.shape[0]):

    branch_inputs = inputs_test[i].reshape(1, n_channels, nx2, nx1) # (bs, c, h, w) = (1, n_channels, nx2, nx1)

    # neval = nx1*nx2
    all_evaluation_points = locations
    if use_fourier_features == False:
        trunk_inputs = all_evaluation_points.unsqueeze(0) # (bs, neval, 2) = (1, nx1*nx2, 2)
    else:
        trunk_inputs = fourier_features(x1_locations, x2_locations, n_fourier).unsqueeze(0)  # (bs, neval, 2+(4*n_fourier)) = (1, nx1*nx2, 2+(4*n_fourier))

    prediction_i = model(branch_inputs, trunk_inputs) # (bs, neval) = (1, nx1*nx2)
    target_i = outputs_test[i].reshape(1, -1)
    
    r2score_i = metrics.r2_score(target_i.flatten().cpu().detach().numpy(), prediction_i.flatten().cpu().detach().numpy()) 
    r2score_list.append(r2score_i)
    
    mse_i = F.mse_loss(prediction_i.cpu(), target_i.cpu())
    mse_list.append(mse_i.item())
    

    if (i+1) % 20 == 0:
        print('TEST SAMPLE '+str(i+1))

        a = inputs_test[ i , : ].reshape(-1,1).cpu().detach().numpy()
        u_truth = outputs_test[i, :].reshape(-1,1).cpu().detach().numpy()
        u_prediction = prediction_i.reshape(-1,1).cpu().detach().numpy()
        
        r2score = metrics.r2_score(u_truth.flatten(), u_prediction.flatten()) 
        relerror = np.linalg.norm(u_truth.flatten() - u_prediction.flatten()) / np.linalg.norm(u_truth.flatten())
        r2score = float('%.4f'%r2score)
        relerror = float('%.4f'%relerror)
        print('Rel. L2 Error = '+str(relerror)+', R2 score = '+str(r2score))

        # Initialize the plot
        fig = plt.figure(figsize=(15,3.5))
        plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.5, wspace = 0.4, hspace = 0.1)

        try:
            ax1.lines.remove(c1[0])
            ax2.lines.remove(c2[0])
            ax3.lines.remove(c3[0])
            ax4.lines.remove(c4[0])

        except:
            pass

        ax1 = fig.add_subplot(1, 4, 1)
        c1 = ax1.contourf( x1_locations.reshape((nx2,nx1)).cpu().detach().numpy(),
                           x2_locations.reshape((nx2,nx1)).cpu().detach().numpy(),
                           np.log(a).reshape((nx2,nx1)), levels=100, cmap=cmap) # set levels automatically
        # This is the fix for the white lines between contour levels (https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills)
        for j in c1.collections:
            j.set_edgecolor("face")
        plt.colorbar(c1)
        plt.title('log(Conductivity field)', fontsize=14)
        plt.xlabel(r'$x_1$', fontsize=12)
        plt.ylabel(r'$x_2$', fontsize=12)


        ax2 = fig.add_subplot(1, 4, 2)
        c2 = ax2.contourf( x1_locations.reshape((nx2,nx1)).cpu().detach().numpy(),
                           x2_locations.reshape((nx2,nx1)).cpu().detach().numpy(),
                           u_truth.reshape((nx2,nx1)), levels=100, cmap=cmap) # set levels as previous levels
        # This is the fix for the white lines between contour levels (https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills)
        for j in c2.collections:
            j.set_edgecolor("face")
        cbar = fig.colorbar(c2)
        plt.title('True Temperature field',fontsize=14)
        plt.xlabel(r'$x_1$', fontsize=12)
        plt.ylabel(r'$x_2$', fontsize=12)

        ax3 = fig.add_subplot(1, 4, 3)
        c3 = ax3.contourf( x1_locations.reshape((nx2,nx1)).cpu().detach().numpy(),
                           x2_locations.reshape((nx2,nx1)).cpu().detach().numpy(),
                           u_prediction.reshape((nx2,nx1)), levels=100, cmap=cmap) # set levels as previous levels
        # This is the fix for the white lines between contour levels (https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills)
        for j in c3.collections:
            j.set_edgecolor("face")
        cbar = fig.colorbar(c3)
        plt.title('Predicted Temperature  field', fontsize=14)
        plt.xlabel(r'$x_1$', fontsize=12)
        plt.ylabel(r'$x_2$', fontsize=12)
        
        ax4 = fig.add_subplot(1, 4, 4)
        c4 = ax4.contourf( x1_locations.reshape((nx2,nx1)).cpu().detach().numpy(),
                           x2_locations.reshape((nx2,nx1)).cpu().detach().numpy(),
                           np.abs(u_truth.reshape((nx2,nx1)) - u_prediction.reshape((nx2,nx1))), levels=100, cmap=cmap) # set levels as previous levels
        # This is the fix for the white lines between contour levels (https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills)
        for j in c4.collections:
            j.set_edgecolor("face")
        cbar = fig.colorbar(c4)
        plt.title('Absolute error', fontsize=14)
        plt.xlabel(r'$x_1$', fontsize=12)
        plt.ylabel(r'$x_2$', fontsize=12)

        plt.tight_layout()
        if save == True:
            plt.savefig(os.path.join(resultdir,'Test_Sample_'+str(i+1)+'.pdf')) 
            plt.close()
        if save == False:
            plt.show()
        print('*'*100)
        
mean_r2score = sum(r2score_list) / len(r2score_list)
print("Mean R2 score Test:\n", mean_r2score)

mse = sum(mse_list) / len(mse_list)
print("Mean Squared Error Test:\n", mse)




print("Time (sec) to complete:\n" +str(finish)) # time for network to train
if save == True:
    print ("------END------")
    sys.stdout = orig_stdout
    q.close()










