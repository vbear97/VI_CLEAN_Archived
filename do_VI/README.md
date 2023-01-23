do_VI contains necessary code to apply black box variational inference to the 1 factor CFA model. 

var_dist.py || Implements variational distributions $q_{\phi}(\theta)$. 

single_cfa_inputs.py || Initialises required inputs for the main function do_vi(): imports statistical data $\mathbb{x}$ from Holzinger and Swineford (1939) dataset, initialises prior distribution hyperparameters, etc. 

single_cfa_model.py || Defines the object sem_model(), which encodes the prior distribution $p(\theta) $ and likelihood distribution $p(\theta | x) $of the 1 factor CFA model, PLUS variational distribution $q_{\phi}(\theta)$ that I have decided to use to approximate the intractable posterior distribution $p(\theta| x)$. The sem_model object holds variational parameters $\phi$, which are optimized in the variational inference algorithm. 

single_cfa_optim.py || Code required to optimize an sem_model object. The primary function is do_vi(), which allows the user to: 

-Specify the type of variational inference algorithm to apply (this is based on what type of divergence measure they choose to minimise)
-Training configurations, such as: 

-Learning Rate 

-Maximum number of iterations 

-Whether or not to use a dynamic learning rate scheduler 

-Starting point 

etc. 

In theory, when trying to compare the performance of different types of variational inference algorithms, I do not adjust the training configurations (to allow ease of comparison) - so most of the training configurations have a default option. 

There are two main files: 

single_cfa_main.py || Provides example of how to apply the standard type of variational inference (based on Kullback Liebler Divergence and the ELBO objective function).

single_cfa_stochastic.py || Provides example of how to apply Renyi Divergence Variational Inference (based on the Renyi Divergence and the Variational Renyi Bound objective function).
