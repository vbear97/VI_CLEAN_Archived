#%%
#This file initialises variational distribution objects, whose parameters are optimised under the Variational Inference algorithm. 

#Variational distribution objects are prefixed with qvar, and are wrapper objects are torch distribution objects. Their main functions are the following: 

#1. Any qvar object is a wrapper object around a torch distributoin object, e.g. normal or gamma. 
#2. The qvar object has an attribute var_params, which are the parameters for this distribution. 
#3. var_params is a tensor with requires_grad = TRUE. This allows us to take the gradient of the VR bound (when doing Renyi Divergence Variational Inference) with respect to the varational parameters, which is the whole point of any black box variational inference algorithm.
#4. qvar has 2 key functions: 
### Sample from the torch distribution object 
### Compute the log_probability. 

# %%
#Import Packages 
import torch
from torch.distributions import Normal, Gamma, Binomial
from torch.distributions import MultivariateNormal as mvn
#hard-coded
offset = 1.0  #offset for the inverse gamma distribution 
##set default torch data type 
torch.set_default_dtype(torch.float64)

# %%
#Multivariate Normal Variational Distribution, with independent components 
class qvar_normal():
    def __init__(self, size, mu=None, log_s=None):
        # we take log_standard deviation instead of standard deviation. This allows for unconstrained optimisation.
        if mu is None:
            mu = torch.randn(size)
        if log_s is None:
            log_s = torch.randn(size)  # log of the standard deviation
        # Variational parameters
        self.var_params = torch.stack([mu, log_s]) #are always unconstrained
        self.var_params.requires_grad = True
    def dist(self):
        return torch.distributions.Normal(self.var_params[0], self.var_params[1].exp())
    def rsample(self, n=torch.Size([])):
        return self.dist().rsample(n)
    def log_prob(self, x):
        return self.dist().log_prob(x).sum()
# %%
#Univariate Inverse Gamma Distribution 
class InverseGamma():
    r'''
    Creates a one dimensional Inv-Gamma Distribution parameterised by concentration and rate, where: 

    X ~ Gamma(concentration, rate)
    Y = 1/X ~  InvGamma(concentration, rate)

    Args: 
    concentration, rate (float or Tensor): concentration, rate of the Gamma distribution
    '''
    def __init__(self, concentration, rate, validate_args = None): 
        self.base_dist = Gamma(concentration = concentration, rate = rate, validate_args=None)
        self.concentration = concentration
        self.rate = rate
    
    def log_prob(self,y):
        #1/0 not a problem here, since log_prob will only be evaluated on theta_sample
        abs_dj = torch.square(torch.reciprocal(y))
        y_rec = torch.reciprocal(y)
        return self.base_dist.log_prob(y_rec) + torch.log(abs_dj)
    
    def rsample(self, n= torch.Size([])): 
        #note that 1/0 is not a problem here.
        base_sample = self.base_dist.rsample(n)
        return torch.reciprocal(base_sample)

# %%
#Multivariate Inverse Gamma Variational Distribution with Independent Components 
class qvar_invgamma():
    def __init__(self, size, alpha=None, beta=None):
        if alpha is None:
            alpha = torch.rand(size) + offset #log_alpha
        if beta is None:
            beta = torch.rand(size) + offset #log_beta
        # Variational parameters
        self.var_params = torch.stack([alpha, beta]) #are always unconstrained
        self.var_params.requires_grad = True
    def dist(self):
        return InverseGamma(concentration= torch.exp(self.var_params[0]), rate= torch.exp(self.var_params[1]))
    def rsample(self, n = torch.Size([])):
        return self.dist().rsample(n)
    def log_prob(self,x):
        return self.dist().log_prob(x).sum() #assuming independent components

# %%
#Degenerate objects are used if we wish to fix certain values. Used only for code testing.
class qvar_degenerate():
    def __init__(self, values):
        self.var_params = values
        self.var_params.requires_grad = False #do not update this parameter
    def dist(self):
        return "Degenerate, see values attribute"
    def rsample(self):
        return self.var_params.clone()
    def log_prob(self, x):
        return torch.tensor([0.0])