# %%
#This file defines the function dovi(), which takes in as user input: 

#1. hyper - hyperparameters for prior distributions of the single factor CFA model 
#2. y_data - y data for single factor CFA model 
#3. degenerate - dictionary of degenerate model parameters in single factor CFA model 
#4. p  - dictionary of variational parameters phi in single factor CFA model 
#5. ometa - dictionary of optimisation metadata values. Best to keep at default: 
#miter - maximum number of training loop iterations 
#lr_nl: learning rate for the nu, lambda model parameters 
#lr_ps: learning rate for the psi, sigma model parameters 
#lr_eta: learning rate for the eta model parameters 
#thresh + patience: once relative error in variational parameters dips below thresh for a consecutive number of iterations (measured by patience), stop training 
#6. div - choice of divergence that we wish to implement. 
#7. K - number of times to sample from variational distribution q when calculating the ELBO or the biased VR bound. 
#8. val (default TRUE) - to calculate validation loss for learning rate scheduler 
#9. lrpatience - learning rate patience for the dynamic scheduler
#10. prefix for filname of optimized sem_model object after training loop. 

#After running dovi, the user gets back: 
#1. optimised sem_model object, with variational parameters that have been optimized under the algorithm (elbo or vr_biased) specified. 

#In addition, the user can view the convergence data on tensorboard. The optimized sem_model object is also saved as a pickled object.
#%%
#For Variational Inference 
import torch

#My packages
from single_cfa_model import *
#Tensorboard 

from single_cfa_benchmarkobjects import VAR

#For Visualisation and Sampling 
import numpy as np
import pandas as pd

#for data storage
import pickle
#for time keeping purposes and progress tracking 
from timeit import default_timer as timer
import copy
from tqdm import trange

#to monitor algorithm convergence 
from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb

torch.set_default_dtype(torch.float64)

#filepath to store tensorboard runs and the pickled sem_model object. 

runspath = '/Users/vivbear/Python_Projects/vi-impl/VI_CLEAN/single_cfa_runs/'
sempath = '/Users/vivbear/Python_Projects/vi-impl/VI_CLEAN/single_cfa_semobjects/'

# %%
#
NUM_SAMP = 60000 #number of times to sample from qvar to create an approximate density

# %%
def dovi(hyper, y_data, degenerate, p, ometa= {'miter': 20000, 'lr_nl': 0.01, 'lr_ps': 0.1, 'lr_eta': 0.1, 'thresh': 10e-4, 'patience':100}, div = 'kl', K = None, aorn = None, val= True, file_comment= '', lrpatience = 1000): 
   
   #initialise sem_model object 
    mysem = sem_model(hyper = hyper, y_data = y_data, degenerate = degenerate, p = p)

    #Initialise optimizer objects
    myrel_error = []
    optimizer = torch.optim.Adam([{'params': [mysem.qvar['nu'].var_params, mysem.qvar['lam'].var_params], 'lr': ometa['lr_nl']},\
     {'params': [mysem.qvar['psi'].var_params, mysem.qvar['sig2'].var_params], 'lr': ometa['lr_ps']},\
         {'params':[mysem.qvar['eta'].var_params], 'lr': ometa['lr_eta']} 
         ])
    
    #dynamic scheduler changes learning rate when change in relative error is low 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience= lrpatience, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    
    #Create a filename 
    filename= file_comment + str(div) + str(aorn) + 'k' + str(K)
    #Create an identifier (nicer file name)
    label = ''
    if div== 'kl' or 'elbo_multi':
        label += 'KL, '
    else:
        label += 'alpha = ' + aorn + ', '
    label+= 'K=  '+ str(K)

    if val:
        filename = filename + 'withscheduler'
    
    #store convergence data in tensorboard
    writer = SummaryWriter(runspath + filename)
  
    
   #prepare optimisation hyperparameters and optimisation objects
    thresh = ometa['thresh']
    patience = ometa['patience']
    iters = trange(ometa['miter'], mininterval = 1)
    tcount = 0
    stime = timer()

    #do optimisation
    for t in iters:
        #First, update gradients
        optimizer.zero_grad()
        if div == 'kl':
            loss = -mysem.elbo()
            val_loss = loss.item()
            loss.backward()
            writer.add_scalar("elbo", val_loss, global_step = t)
        elif div == 'elbo_multi':
            loss = -mysem.elbo_multi(K = K)
            val_loss = loss.item()
            loss.backward()
        elif div == 'vr_biased':
            loss = mysem.vr_biased(K = K, a = aorn)
            val_loss = loss.item()
            loss.backward()

        optimizer.step()

        #Record values to tensorboard
        writer.add_scalars("variational parameters", mysem.scalars(), global_step= t) 

        #To do or not do validation?

        if val:
            #record validation loss
            writer.add_scalar('val_loss', scalar_value = val_loss, global_step = t)
            #do the step for val loss 
            scheduler.step(val_loss) 
    
        #is there convergence?
        next = mysem.ev()
        if (t>1):
            error = rel_error(prev = prev, next = next)
            myrel_error.append(error) #record relative error
            # print("error = ", error)
            if(error <= thresh):
                tcount+=1
                if (tcount >= patience):
                    print("Converged at t=", t)
                    break
            else:
                tcount = 0
    
        prev = next
    
    #loop completed
    etime = timer()
    writer.close()

    #update convergence and optimisation metadata
    
    cmeta = {'rel_error': myrel_error,'ctime': etime - stime, 'iter_ran': t, 'div': div, 'K': K, 'aorn': aorn, 'val': val, 'learning rate patience': lrpatience,'p': p, 'filename':filename, 'label' : label}

    mysem.update_meta(cmeta = copy.deepcopy(cmeta), ometa = copy.deepcopy(ometa)) 

    #Sample from the optimised variational distributions 
    var = ['nu.1', 'nu.2', 'nu.3', 'lam.1', 'lam.2', 'psi.1', 'psi.2', 'psi.3', 'sig2']
    num_sample = torch.tensor([NUM_SAMP])
    vb_sample = np.concatenate([mysem.qvar[key].dist().rsample(num_sample).detach().numpy() for key in mysem.qvar if key!= 'eta'], axis = 1)
    vbdf = pd.DataFrame(vb_sample, columns = var)

    mysem.update_datasample(vbdf)

    print("log directory name", filename)
    
    #pickle the sem_model object
    with open(sempath + filename + '.pickle', 'wb') as f:
        pickle.dump(mysem, f)

    return mysem
