This repository provides all Python code required to apply black box Renyi Divergence Variational Inference to a single factor Confirmatory Factor Analysis (CFA) model with mean-field assumption, as investigated in my master's thesis. The observed data used is visual perception tests results from the the Holzinger and Swineford (1939) dataset. 

NOTE: As of Jan 2023, efforts are being made to develop code for applying black box Renyi Divergence VI to more complex Structural Equation Models (SEM). The single factor CFA is the most elementary of SEM models. 

Thesis: An Application of Renyi Divergence Variational Inference to Structural Equation Modelling. 

As of Jan 2023, the repo has 2 parts, do_VI and eval_VI.

Part 1: do_VI 
################

The code required to apply black box Variational Inference to the single factor CFA model. Two key objects are used: 

-qvar objects

-sem_model object 

A qvar object is basically a wrapper around a Pytorch distribution object, and represents the variational distribution $q_{\phi}(\theta)$ on the model parameters $\theta$. Any qvar object does two things: 

a. Holds the variational parameters $\phi$, which are stored as gradient-tracking tensors.

b. Given the variational parameters $\phi$, and the qvar object's innate Pytorch distribution (e.g. Normal, Inverse-Gamma), the qvar object produces a random sample of $\theta$ realisations from the resultant $q_{\phi}(\theta)$ distribution. My thesis implements VI by using reparameterisation gradients, so all qvar objects use the .rsample command found in Pytorch.

An sem_model object contains the two essential ingredients for applying Variational Inference to the 1 factor Bayesian CFA model: 
1. Log-Likelihood and Prior
Given the current value of CFA model parameters $\theta$ (stored as an attribute), calculates log-likelihood and log-prior for the 1 factor CFA model

2. Entropy and Variational Distributions 
The sem_model object contains, as an attribute, a sequence of qvar objects which encodes the structure of the variational distributions we wish to use, to approximate the true, uncomputable posterior distribution $p(\theta | x)$. For the 1 factor CFA model, we use a mean field assumption (i.e. all components of the model parameter $\theta$ are independent), with component-wise distributions being either Inverse Gamma or Normal. 

Given 1. and 2, an sem_model object takes in data $x$ and outputs the value of a scalar, biased Renyi objective function (user specified, depending on what type of VI algorithm we choose and which divergence measure we want to minimise). VI works by minimising this objective function with respect to the variational parameters (stored as tensors in qvar objects). 

After optimisation, the sem_model object (and its optimised variational qvar attribute) is used for Bayesian inference.

################################
Part 2: eval_VI

The code required to evaluate and visualise the performance of a black box Variational Inference algorithm. This is done by comparing the optimized variational distribution with MCMC approximations of the posterior. 

