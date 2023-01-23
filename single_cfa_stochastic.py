#%%
#This is a script to mod out randomness in stochastic optimisation when K is small. 
#For each of the stochastic optimization resulting in q^_s, s = 1...sruns, evaluate the evaluation metrics and then average those. 
#Do not change initialisation of variational parameters in between runs. 
#Question: do we observe lots of stochastic optimisation?
import torch 
torch.set_default_dtype(torch.float64)
from single_cfa_inputs import *
from single_cfa_model import *
from single_cfa_optim import *
import pickle

# %%
#We will try for K = 10, alpha = 0.5, -1, -5, 10 
runs = 10 
# %%
alpha = 0.5 
a5 = {}
for i in range(runs):
    a5[i]= dovi(hyper = hyper, y_data = y_data, degenerate = degenerate, p = p, div = 'vr_biased', K = 10, aorn = alpha, file_comment= 'stochastic_run_' + str(i)+ ' ')
# %%
