This repository provides all Python code required to apply black box Renyi Divergence Variational Inference to a single factor Confirmatory Factor Analysis (CFA) model with mean-field assumption, as investigated in my master's thesis. The observed data used is visual perception tests results from the the Holzinger and Swineford (1939) dataset. 

NOTE: As of Jan 2023, efforts are being made to develop code for applying black box Renyi Divergence VI to more complex Structural Equation Models (SEM). The single factor CFA is the most elementary of SEM models. 

Thesis: An Application of Renyi Divergence Variational Inference to Structural Equation Modelling. 

As of Jan 2023, the repo has 3 parts: 
1. do_VI 
The code required to apply black box Variational Inference to the single factor CFA model. Two key objects are used: 

-qvar objects
-sem_model object 

A qvar object is basically a wrapper around a Pytorch distribution object, and represents the variational distribution $q_{\phi}(\theta)$ on the model parameters $\theta$. Any qvar object does two things: 

a. Holds the variational parameters $\phi$, which are stored as gradient-tracking tensors.
b. Given the variational parameters $\phi$, and the qvar object's innate Pytorch distribution (e.g. Normal, Inverse-Gamma), the qvar object produces a random sample of $\theta$ realisations from the resultant $q_{\phi}(\theta)$ distribution. My thesis implements VI by using reparameterisation gradients, so all qvar objects use the .rsample command found in Pytorch.

The sem_model object is a big wrapper object that holds all of the information and functions necessary to perform VI on a single factor CFA. It has the following inputs: 

a. y_data 
b. 

a. Holds the variational parameter vector $\phi$, which consists of a collection of qvar objects. 
b. The function self.theta_sample() produces a reparameterised random sample of $\theta$ from the variational distribution $q_{\phi}(\theta)$. In my thesis, the variational family chosen specifies pairwise independence for all model parameters.
c. The functions self.log_like(theta_sample) and self.log_prior(theta_sample) takes in a random sample generated from $q_{\phi}(\theta)$ and uses the current value of the variational parameters held in sem_model to compute a value of the log likelihood p 


2. eval_VI
The code required to evaluate the performance of a black box Variational Inference procedure. 

3. vis_VI 

The code required to visualise the performance of a black box Variational Inference procedure.
