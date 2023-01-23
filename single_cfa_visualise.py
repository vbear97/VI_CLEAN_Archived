#%%
#This file visualises the results of a VI routine for the single CFA model for both non-latent model parameters and latent model parameters.

#Visualisation is usually done by comparing is done by comparing the approximate posteriors produced by the VI algorithm, to the approximate posteriors produced by the MCMC routine. 

#Functions for visualising results on non-latent model parameters :

#Intermediate Functions (not for end-user use)
#raw_plot_dens 
#raw_plot_credint

#Visualisation Functions (for end user use)
#vis_all()

#Functions for visualising results on latent model parameters: 

#plot_etameans
#plot_etavar 

# %%
#Import packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from single_cfa_benchmarkobjects import *
import torch 
# %%
#Set a colour dictionary to visualise results 
#Default Colours 
COLOURS = ['blue', 'orange', 'green','purple', 'brown', 'pink', 'yellow', 'olive', 'cyan', 'magenta']
VAR = ['nu.1', 'nu.2', 'nu.3','lam.2', 'lam.3', 'psi.1', 'psi.2', 'psi.3', 'sig2']

#Reserved colours 
#red = mcmc
# %%
#Functions prefixed with raw_ are intermediate functions and not for end_user use.

#Plot densities 

def raw_plot_dens(data, title = 'Non-Latent Parameters in Single CFA Model: Comparison of Estimated Posterior Densities', figsize = (10,10), yeskde = True):

    #Draw figure and axes 
    fig, ax = plt.subplots(5,2, constrained_layout = True, figsize = figsize)
    fig.delaxes(ax[4,1])
    fig.suptitle(title)

    #autocreate handles for legend
    handles = [mpatches.Patch(color = data[key][1], label = key) for key in data]
    fig.legend(handles= handles, loc = 'lower right')

    #for v,a in zip(var,ax.flatten()):
    for v,a in zip(VAR,ax.flatten()):
        if yeskde:
            for key in data:
                sns.kdeplot(data = data[key][0][v], color = data[key][1], ax = a)
        else:
            for key in data:
                sns.kdeplot(data = data[key][0][v], color = data[key][1], ax = a, stat = 'density', kde = True, bins = 100)
# %%

#Plot MCMC credible intervals.

def raw_plot_credint(data, q1 = 0.0275, q2 = 0.975, figsize = (10,10), title = ' Non-Latent Parameters in Single CFA Model : Comparison of Credible Interval Widths from Estimated Posterior Desntities'):
    #Draw figure and axes 
    fig, ax = plt.subplots(5,2, constrained_layout = True, figsize = figsize)#harded coded, not dynamic if we change the size of M
    fig.delaxes(ax[4,1])
    #add title
    fig.suptitle(title)

    #add legend 
    handles = [mpatches.Patch(color = data[key][1], label = key) for key in data]
    if ('MCMC' in data.keys()):
        handles.append(mpatches.Patch(color = 'red', ls = '--', label = 'MCMC Mean')) #MCMC mean

    fig.legend(handles= handles, loc = 'lower right')

    quantiles = {key: [data[key][0].quantile([q1, q2]), data[key][0].mean()] for key in data}

#plot as credible intervals
    for v,a in zip(VAR,ax.flatten()):
        for key, y in zip(data, range(len(data))):
            #plot credible interval
            color = data[key][1]
            a.plot(quantiles[key][0][v], (y,y), color = color, linewidth = 10, alpha = 0.5)
            #plot mean
            a.plot(quantiles[key][1][v], y, color = 'black', marker = 'o')
        #then, plot mcmc mean as a reference line 
        if 'MCMC' in data.keys():
            a.axvline(quantiles['MCMC'][1][v], color = 'red', ls = '--')
        #add credint reference lines for chosen 
        # if refcredint in data.keys():
        #     a.axvline(quantiles[refcredint][0][v][q1], color = data[refcredint][1], ls = '--')
        #     a.axvline(quantiles[refcredint][0][v][q2], color = data[refcredint][1], ls = '--')
        
        # if refmean in data.keys():
        #     a.axvline(quantiles[refcredint][1][v], color = 'black', ls = '--')

        #title, cosmetic issues
        a.set_title(label = v)
        a.yaxis.set_visible(False)

        #xaxis tick marks

# %%
def vis_all(dsems, figsize = (10,10), q1 = 0.025, q2 = 0.975):
    '''
    Takes in a dictionary of optimised sem_model objects, plots the estimated posterior densities of non-latent model parameters against those obtained by MCMC and MFVB. 
    
    MCMC estimates are coloured black.
    MFVB estimates are coloured red.
    '''
    if len(dsems.keys()) > len(COLOURS):
        print("Error: not enough colors")
        return

    #Create a dictionary: 
    #label: [sem_model object, colour (as string)]

    col = COLOURS[0:len(dsems.keys())]
    data = {dsems[k].cmeta['label']: [dsems[k].datasample, c] for k, c in zip(dsems, col)}

    #Add in reference objects 
    data['MCMC'] = [single_cfa_mcmc.datasample, 'red']
    #data['Mean Field Variational Bayes'] = [mfvb.datasample, 'black']

    raw_plot_credint(data = data)
    raw_plot_dens(data = data)

    return

# %%
#The following functions are for visualising results of VI routine for latent model parameters compared to MCMC. 

def plot_etameans(sem_model,  figsize = (5,5), title = 'Scatterplot Comparing Eta Means'):
    ''''
    Takes in an sem_model object and returns scatterplot of VI eta means (y axis) vs. MCMC eta means (x axis). 
    '''
    #Generate Eta Data
    vbeta = sem_model.qvar['eta'].var_params[0].detach().numpy()
    filter_eta = [col for col in mcobj.model if col.startswith('eta.')]
    mceta = mcobj.model[filter_eta].mean()
    #

    fig, ax = plt.subplots(figsize = figsize)
    fig.suptitle(title)

    ax.scatter(x = mceta, y = vbeta)
    ax.set_ylabel('VI Eta Means')
    ax.set_xlabel('MCMC Eta Means')
    ax.axline(xy1 = (0,0), slope = 1)
# %%
def plot_etavar(sem_model, figsize = (5,5), title = 'Scatterplot Comparing Eta Variances'):

    '''
    Takes in an sem_model object and returns scatterplot of VI eta variances (y axis) vs. MCMC eta variances (y axis).

    '''
    vbeta = torch.square(sem_model.qvar['eta'].var_params[1].exp()).detach().numpy()
    filter_eta = [col for col in mcobj.model if col.startswith('eta.')]
    mceta = mcobj.model[filter_eta].var()
    
    fig, ax = plt.subplots(figsize = figsize)
    ax.set_ylim([np.amin(vbeta), np.amax(vbeta)])
    ax.set_xlim([np.amin(mceta), np.amax(mceta)])
    ax.scatter(x = mceta, y = vbeta)
    ax.set_ylabel('VR-Alpha Variances')
    ax.set_xlabel('MCMC Variances')
    ax.axline(xy1 = (0,0), slope = 1)