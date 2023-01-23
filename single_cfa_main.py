# %%
#This is the main file to obtain optimised variational distributions for the single factor CFA model. 
#%%
#set default 
import torch 
torch.set_default_dtype(torch.float64)
from single_cfa_inputs import *
from single_cfa_model import *
from single_cfa_optim import *

# %%
#all working
kl = dovi(hyper = hyper, y_data = y_data, degenerate = degenerate, p = p, div = 'elbo_multi', K = 10)

# %%
