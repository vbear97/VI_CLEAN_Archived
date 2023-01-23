# %%
#This file defines the object single_cfa_model which takes as input: 

#1. y_data from single factor CFA model. This is usually the Holzinger-Swineford dataset downloaded in the single_cfa_inputs file. 
#2. degenerate. This is if (for testing purposes) we wish for some of the single factor CFA model parameters to be fixed. 
#3. hyper. Hyperparameters for the prior distributions for the single factor CFA model, which is initialised in the single_cfa_inputs file. 
#4. p. These are the initialised starting points for variational parameters.

#Default inputs are also included, rarely changed:
#5. cmeta. This standards for "convergence metadata". 
#6. ometa. This standards for 

#The distributional family and structure of the mean-field variational family qvar, log-likelihood and log-prior are all hard coded into the single_cfa_model. Specifically, the log-likelihood and log-prior are as described in the paper, and the variational family uses a mean-field assumption (i.e. components are pairwise independent), with the distributions as described in paper. 

#Given this hard-coded information, the single_cfa model performs the following key functions: 

#1. Given current value of variational parameters, computes ELBO. This is given by the function self.elbo()
#2. Given current value of variational parameters, computes the biased VR bound. This is given by the function self.vr_biased(a, K) which takes in an argument of alpha (a) and K. 

#%%
import torch
from torch.distributions import Normal, Gamma, Binomial
from torch.distributions import MultivariateNormal as mvn
from torch.distributions import Categorical as cat 
import pandas as pd
import copy

#import variational distributions 
from var_dist import *
##set default torch data type 
torch.set_default_dtype(torch.float64)
    
class sem_model():
    def __init__(self, y_data, degenerate, hyper, p = {}, cmeta= {"rel_error": None, "ctime": None, "max_iter": None}, ometa= {}, datasample = None):
        
        #user input:
        self.y_data = y_data.clone().detach()
        self.degenerate = copy.deepcopy(degenerate)
        self.hyper = copy.deepcopy(hyper)
        self.n = self.y_data.size(0) 
        self.m = self.y_data.size(1)
        self.lam1 = torch.tensor([1.0])
        self.datasample = datasample
        
        #starting point 

        #clone p to make sure there are absolutely no aliasing issues
        p = copy.deepcopy(p)
        self.p = p

        #update information about variational family 
        self.qvar = {'nu': qvar_normal(self.m, mu = p.get('nu.m'), log_s = p.get('nu.ls')),\
            'lam': qvar_normal((self.m)-1, mu =p.get('lam.m'), log_s = p.get('lam.ls')),\
            'eta': qvar_normal(self.n, mu = p.get('eta.m'), log_s = p.get('eta.ls')),\
            'psi': qvar_invgamma(self.m, alpha = p.get('psi.a'), beta = p.get('psi.b')),\
            'sig2': qvar_invgamma(1, alpha = p.get('sig2.a'), beta = p.get('sig2.b'))}
        self.qvar.update(self.degenerate)

        #convergence metadata 
        self.cmeta = cmeta
        self.ometa = ometa

    def generate_theta_sample(self):
        #generate a single sample of cfa model parameters theta from the current variational distribution qvar.
        return {var: qvar.rsample() for (var,qvar) in self.qvar.items()}

    def log_like(self,theta_sample):
        #given a value of theta, calculate the cfa model log-likelihood.
        like_dist_cov = torch.diag(theta_sample['psi'])

        lam_full = torch.cat((self.lam1, theta_sample['lam']))

        like_dist_means = torch.matmul(theta_sample['eta'].unsqueeze(1), lam_full.unsqueeze(0)) + theta_sample['nu']
        
        log_like = mvn(like_dist_means, covariance_matrix= like_dist_cov).log_prob(self.y_data).sum()

        #print("log_like =" ,log_like)

        return log_like

    def log_prior(self, theta_sample): 
        #given a value of theta, calculate the cfa model log-prior.

        #hard coded prior 
        priors = {'nu': Normal(loc = self.hyper['nu_mean'], scale = torch.sqrt(self.hyper['nu_sig2'])), \
            
        'sig2': InverseGamma(concentration = self.hyper['sig2_shape'], rate = self.hyper['sig2_rate']),\

        'psi': InverseGamma(concentration = self.hyper['psi_shape'], rate= self.hyper['psi_rate']),\

        'eta': Normal(loc = 0, scale = torch.sqrt(theta_sample['sig2'])),\

        'lam': Normal(loc = self.hyper['lam_mean'], \
            scale = torch.sqrt(self.hyper['lam_sig2']*(theta_sample['psi'][1:])))
            }
        
        log_priors = {var: priors[var].log_prob(theta_sample[var]).sum() for var in priors if var not in self.degenerate}

        #print("log_priors", log_priors)

        return sum(log_priors.values())

    def entropy(self, theta_sample):
        #given a value of theta in addition to qvar, calculate entropy.
        qvar_prob = {var: self.qvar[var].log_prob(sample) for (var,sample) in theta_sample.items()}

        #print("entropy", qvar_prob)

        return sum(qvar_prob.values())
    
    def elbo(self):
        #calculate the elbo(), which is used for KL Divergence Variational Inference.
        theta_sample = self.generate_theta_sample()

        # print("theta_sample", theta_sample)
        
        return self.log_like(theta_sample) + self.log_prior(theta_sample) - self.entropy(theta_sample)
    
    def elbo_multi(self, K = 10):
        elbos = torch.stack([self.elbo() for k in range(K)])
        return elbos.mean()

    def vr_biased(self, K = 10, a = 0.5): 
        #calculate the biased vr_bound, which is used for  Renyi Divergence Variational Inference. Technically, only a values larger than 0 correspond to valid divergences. However, the algorithm can be extended to alpha values <0. 

        logw = torch.stack([self.elbo() for k in range(K)])
        logw = (1-a)*logw
        c = logw.max()
        logw_correct = logw - c
        lse = c + torch.logsumexp(logw_correct, dim = 0) #equal to logsumexp(logw), via offsetting
        lse_av = lse - torch.log(torch.tensor(K)) #divide by K to take average.
        vr_bound = lse_av/(1-a)

        #if (a > 0): 
            #loss = -vr_bound
        #else:#if a < 0
            #loss = vr_bound
        loss = -vr_bound
        return loss

    def scalars(self): 
        scalars =\
                    {'nu1_sig': self.qvar['nu'].var_params[1][0].exp().item(),\
                    'nu2_sig': self.qvar['nu'].var_params[1][1].exp().item(), \
                    'nu3_sig': self.qvar['nu'].var_params[1][2].exp().item(),\
                    'nu1_mean': self.qvar['nu'].var_params[0][0].item(),\
                    'nu2_mean': self.qvar['nu'].var_params[0][1].item(), \
                    'nu3_mean ': self.qvar['nu'].var_params[0][2].item(), \
                    'lambda2_mean': self.qvar['lam'].var_params[0][0].item(),\
                    'lambda2_sig': self.qvar['lam'].var_params[1][0].exp().item(),\
                    'lambda3_sig': self.qvar['lam'].var_params[1][1].exp().item(),\
                    'lambda3_mean': self.qvar['lam'].var_params[0][1].item(),\
                    'psi_1_alpha': self.qvar['psi'].var_params[0][0].exp().item(),\
                    'psi_2_alpha': self.qvar['psi'].var_params[0][1].exp().item(),\
                    'psi_3_alpha': self.qvar['psi'].var_params[0][2].exp().item(),\
                    'psi_1_beta': self.qvar['psi'].var_params[1][0].exp().item(), \
                    'psi_2_beta': self.qvar['psi'].var_params[1][1].exp().item(), \
                    'psi_3_beta': self.qvar['psi'].var_params[1][2].exp().item(), \
                    'sig2_alpha': self.qvar['sig2'].var_params[0].exp().item(),\
                    'sig2_beta': self.qvar['sig2'].var_params[1].exp().item()}
        return scalars

    def ev(self):
        var_params = {key:self.qvar[key].var_params.clone().detach() for key in self.qvar if key not in self.degenerate}

        return var_params
    
    def update_meta(self, cmeta, ometa):
        '''keys: rel_error, max_iter, ctime'''
        self.cmeta = cmeta
        self.ometa = ometa
        return
    
    def update_datasample(self, datasample):
        '''
        Datasample is a pandas dataframe
        '''
        self.datasample = datasample
        return

#other functions
    
def rel_error(prev, next):
    '''
    dict j : sem_model.var_params. key:torch.tensor entries. assumed same keys.
    '''
    rel = {key: ((next[key] - prev[key])/prev[key]).abs().max() for key in prev}
    return max(rel.values())

def getparams(sem_model):
    params = {}
    label =  ['nu.1.mu', 'nu.2.mu', 'nu.3.mu']
    value = sem_model.ev()['nu'][0].clone()
    for l, v in zip(label, value):
        params[l] = v.item()

    label = ['lam.1.mu', 'lam.2.mu']
    value = sem_model.ev()['lam'][0].clone()
    for l, v in zip(label, value):
        params[l] = v.item()
    
    label =  ['nu.1.sig', 'nu.2.sig', 'nu.3.sig']
    value = sem_model.ev()['nu'][1].exp().clone()
    for l, v in zip(label, value):
        params[l] = v.item()
    
    label =  ['lam.1.sig', 'lam.2.sig']
    value = sem_model.ev()['lam'][1].exp().clone()
    for l, v in zip(label, value):
        params[l] = v.item()
    
    label =  ['psi.1.a', 'psi.2.a', 'psi.3.a']
    value = sem_model.ev()['psi'][0].exp().clone()
    for l, v in zip(label, value):
        params[l] = v.item()
    
    label =  ['psi.1.b', 'psi.2.b', 'psi.3.b']
    value = sem_model.ev()['psi'][1].exp().clone()
    for l, v in zip(label, value):
        params[l] = v.item()
    
    label =  ['sig2.a']
    value= sem_model.ev()['sig2'][0].exp().clone()
    for l, v in zip(label, value):
        params[l] = v.item()
    
    label =  ['sig2.b']
    value= sem_model.ev()['sig2'][1].exp().clone()
    for l, v in zip(label, value):
        params[l] = v.item()
    
    return params

# %%
