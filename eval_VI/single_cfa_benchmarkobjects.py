#%%

#The following file is to obtain data samples  for non-latent model parameters, from the MCMC and MFVB (mean field variatiaonl Bayes) posterior distributions.

#Posterior approximations produced by the Hamiltonian MCMC algorithm are treated as the gold standard or "ground truth". 

#The MFVB algorithm is an analytical algorithm for implementing KL Divergence Variational inference. Unlike the VI algorithms implemented here, the MFVB algorithm is not black box and is specific to our single factor CFA model.

#The MFVB algorithm is implemented by Dr Khue Dhue Dang in the paper "Fitting Structural Equation Models via Variational Approximations". 

#https://www.tandfonline.com/doi/abs/10.1080/10705511.2022.2053857


##For now, artefact objects are provided.
PATH = '/Users/vivbear/Python_Projects/vi-impl/VI_CLEAN/single_cfa_benchmarks/'
import pickle 
import torch
import numpy as np
import pandas as pd

##By Feb 2023, code will be provided to generate the benchmark MCMC and MFVB distributions.

# %%
VAR = ['nu.1', 'nu.2', 'nu.3', 'lam.1', 'lam.2', 'psi.1', 'psi.2', 'psi.3', 'sig2']
NUMSAMP = 60000
# %%
#Import mcmc dataframe 

#Posterior approximations of non-latent model parameters nu, lambda, psi and sigma are obtained from Hamiltonian Monte Carlo. For each of these posterior approximations, a sample of size 60,000 is obtained and these are put into the dataframe titled "mcdf".

filename = PATH + "mcmc13071hpickle.pickle"
read = "rb"
mcmcpickle = open(filename, read)
fit = pickle.load(mcmcpickle)
fitp = fit.to_frame() 
#has datasamples for all model parameters, INCLUDING latent AND non-latent.
mcdf = fitp[VAR]

#Single dataframe called mcdf 

# %%

#Make wrapper mcmc object called mc_model 

class mcobj:
    def __init__(self, datasample = {}):
        self.datasample = datasample
single_cfa_mcmc = mcobj(datasample = mcdf)


# %%
#Implement Mean Field Variational Bayes algorithm 

#Download mfvb object 

# class dataobj:
#     def __init__(self, datasample):
#         self.datasample = datasample

# filename = PATH + 'mfvb.pickle'
# mfvb = pickle.load(open(filename, 'rb'))

# mfvb_sample = np.concatenate([mfvb[key].rsample(NUMSAMP).detach().numpy() for key in mfvb], axis = 1)
# mfdf = pd.DataFrame(mfvb_sample, columns = VAR)

# mfvb = dataobj(datasample = mfdf)
