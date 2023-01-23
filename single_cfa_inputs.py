# %%
#This file initialises key quantities for the 1 factor Confirmatory Factor Analysis Model: 

#1. Imports y data from Holzinger Swineford (1939) dataset.
#2. Initialises starting point for the variational parameters phi. 
#3. Initialises hyperparameters for prior distributions on the 1 factor CFA model.

#The quantities y_data, p and hyper are used as inputs in the sem object. 

#%%
#Import packages 
import torch
import numpy as np
import pandas as pd
torch.set_default_dtype(torch.float64)

# %%
#3 dimensional y-data for 1 factor CFA model.

#Import first 3 tests results from Holzinger Swineford (1939) Dataset.
from semopy.examples import holzinger39
hdata = holzinger39.get_data() #hdata is pandas dataframe
myhdata = hdata[['x1', 'x2','x3']] #want only visual perception, cubes, lozenges test results, in that order 
y_data = torch.tensor(myhdata.values, requires_grad=False) #convert to  tensor 

# %%
#initialised starting point for phi variational parameters

p = {'nu.m':torch.zeros(y_data.shape[1]), \
    'nu.ls': torch.zeros(y_data.shape[1]),\
    'lam.m': torch.ones(y_data.shape[1]-1),\
    'lam.ls': torch.zeros(y_data.shape[1]-1),
    'psi.a': torch.ones(y_data.shape[1]),\
    'psi.b': torch.ones(y_data.shape[1]),\
    'sig.a': torch.tensor(1.00),\
    'sig.b': torch.tensor(1.00),\
    'eta.m': torch.zeros(y_data.shape[0]),\
    'eta.ls': torch.ones(y_data.shape[0])}
# %%
#Hyperparameters for prior distributions on the 1 factor CFA model

#Set Hyper-parameters 
#sig_2 ~ InvGamma
sig2_shape = torch.tensor([0.5])  
sig2_rate = torch.tensor([0.5])  

#psi ~ iid Inv Gamma for j = 1..m 
psi_shape = torch.tensor([0.5])  
psi_rate = torch.tensor([0.005])  

#nu ~ iid Normal for j = 1...m
nu_sig2 = torch.tensor([100.0])  
nu_mean = torch.tensor([0.0])

#lam_j | psi_j ~ id Normal(mu, sig2*psi_j)
lam_mean = torch.tensor([0.0])
lam_sig2 = torch.tensor([1.0])

#No degenerate prior distributions (degenerate prior distributions are sometimes used for model testing purposes)
degenerate = {}

#Concatenate
hyper = {"sig2_shape": sig2_shape, "sig2_rate": sig2_rate, "psi_shape": psi_shape, "psi_rate": psi_rate, "nu_sig2": nu_sig2, "nu_mean": nu_mean, "lam_mean": lam_mean, "lam_sig2": lam_sig2}


# %%
#