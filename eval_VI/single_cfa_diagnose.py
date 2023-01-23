#This is a file to evaluate the results of the VI algorithms for a single CFA model. 

#Evaluation is done by comparing the approximate posteriors produced by the VI algorithm, to the approximate posteriors produced by the MCMC routine. 

#IN Bayesian statistics, approrpriately tuned MCMC routines are treated as the gold standard. 

#For any given model parameter, both the VI routine and the MCMC routine will produce an approximate posterior distribution. 

# We will refer to the approximate posterior distribution produced by VI algorithm as qvar. 

#To compare the VI algorithm to the MCMC benchmark, we are concerned with three main quantities: 

# DiffMean: Mean of MCMC distribution - Mean of qvar 
# DiffWidth: Width of 95% MCMC credible interval - Width of 95% qvar credible interval 
# Overlap: Amount of overlap between 95% MCMC credible interval and 95% qvar interval.

#Functions for end_user: 

#global_compwidth
#global_compmeans
#global_overlap 

#Each of these three functions takes in a dictionary of sem_model objects.


#%%
#Import packages 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyrsistent import b
import seaborn as sns
import matplotlib.patches as mpatches
from single_cfa_benchmarkobjects import *

# %%
#Intermediate functions - not for end user 

#Calculate lower and upper quantile + cred interval width 

def calc_quant(sem_model, q= [0.025, 0.975]):
    '''
    Takes in an sem_model object, and computes the lower and upper quantiles of estimated distributions for non-latent model parameters.
    '''
    vbdf = sem_model.datasample
    #extract metadetails and name the dataframe
    qdf = vbdf.quantile(q, axis = 0)
    qdf = qdf.T
    #compute width
    qdf['CredWidth'] = qdf[q[1]] - qdf[q[0]]
    return qdf   

# %%
def calc_into(a_start, a_end, b_start, b_end , mode = 'lap'):
    if (a_start > b_end) or (b_start > a_end):
        return None
    else:
        os = max(a_start, b_start)
        oe = min(a_end, b_end)
        if (mode == 'os'):
            return os
        elif (mode == 'oe'):
            return oe
        elif (mode == 'lap'):
            return oe-os

# %%
def calc_compmodels(sem_model, refobj = single_cfa_mcmc, qu = [0.025, 0.975]):
    '''
    Calculates CI width, MCMC width, %overlap, %above and % below  + CI mode minus MC mode. Hard coded - comparison to MCMC credible interval 
    '''
    #rename columns, for convenience
    v = calc_quant(sem_model, qu).rename(columns = {'CredWidth': 'vbcw', qu[0]: 'vlow', qu[1]: 'vhigh'})
    r = calc_quant(refobj,qu).rename(columns = {'CredWidth': 'refcw', qu[0]: 'rlow', qu[1]: 'rhigh'})
    q= pd.concat([v, r], axis = 1)

    #Is there overlap?
    q['has_lap'] = ~((q['vlow'] > q['rhigh']) | (q['rlow'] > q['vhigh']))
    q['oe'] = np.minimum(q['vhigh'], q['rhigh'])
    q['os'] = np.maximum(q['vlow'], q['rlow'])
    q['lap']= (q['oe']- q['os'])*q['has_lap']

    #% Overlap 
    q['%lap'] = q['lap']/q['refcw']

    #% Overlap, as percentage of refcw
    q['%vblap'] = q['lap']/q['refcw']

    #  %High-Lap 
    q['%hlap'] = ((q['vhigh'] - q['oe'])/q['vbcw'])*q['has_lap']
    
    # %Low Lap 
    q['%llap'] = ((q['os'] - q['vlow'])/q['vbcw'])*q['has_lap']

    #% difference in confidence interval width 
    q['%dwidth'] = (q['vbcw'] - q['refcw'])/q['refcw']

    # absolute difference in standard deviation 
    q['vstd'] = sem_model.datasample.std()
    q['rstd'] = refobj.datasample.std()
    q['dstd'] = q['vstd']-q['rstd']

    #%percentage difference in means (central tendency measure)
    q['vmean'] = sem_model.datasample.mean()
    q['rmean']= refobj.datasample.mean()
    q['%dmean'] = (q['vmean'] - q['rmean'])/(q['rmean'])
    #q['absmean']=np.abs(q['vmean'], q['rmean'])

    #%difference in modes
    q['vmode'] = sem_model.datasample.quantile(0.5)
    q['rmode'] = refobj.datasample.quantile(0.5)
    q['%dmode'] = (q['vmode'] - q['rmode'])/q['rmode']
  
    return q
# %%
def calc_all(sem_model, refobj = single_cfa_mcmc):
    '''
    Calculates DiffMean, DiffWidth, Overlap compared to the reference (in this case MCMC) for non-latent model parameters.
    '''
    table = calc_compmodels(sem_model = sem_model, refobj = refobj)
    table = table[['%dmean', '%dwidth', '%lap']]
    return table
# %%

def loc_compmodels(vbsem, refobj = single_cfa_mcmc, qu = [0.025, 0.975], thresh = {'%dw': 0.10, '%lap': 0.90, '%dm' : 0.05}):

    '''
    Wrapper function for calc_comp models. 


    Calculates DiffMean, DiffWidth, Overlap compared to the reference SEM model object (my default, MCMC) for non-latent model parameters. Then, does some colouring.

    '''

    q = calc_compmodels(vbsem = vbsem, refobj = refobj, qu = qu)
    locq = q[['%dwidth', '%dmean', '%lap', '%hlap', '%llap']]

    #styling 
    #highlight means according to threshold 
    #if % diff > 0.05, black #negative information''
    #if %diff < 0.05, blue 

    lq = locq.style

    slice = ['%dwidth']
    lq = lq.applymap(lambda v: 'opacity: 20%;' if (v < thresh['%dw']) and (v > -thresh['%dw']) else None, subset = slice).applymap(lambda v: 'color: black;' if v >=thresh['%dw'] else None, subset = slice).applymap(lambda v: 'color: blue;' if v <= -thresh['%dw'] else None, subset = slice)
    
    # if %lap >= thresh, highlight 'green'
    slice = ['%lap']
    lq = lq.applymap(lambda v: 'opacity: 20%;' if (v < thresh['%lap']) and (v > 0) else None, subset = slice).applymap(lambda v: 'color: red;' if v==0 else None, subset = slice).applymap(lambda v: 'color: green;' if (v>thresh['%lap']) and (v< 1) else None, subset = slice).applymap(lambda v: 'color: orange;' if v==1 else None, subset = slice)

    # # %dmean
    slice = ['%dmean']

    lq = lq.applymap(lambda v: 'opacity: 20%;' if (v < thresh['%dm']) and (v > -thresh['%dm']) else None, subset = slice).applymap(lambda v: 'color: black;' if v >=thresh['%dm'] else None, subset = slice).applymap(lambda v: 'color: blue;' if v < -thresh['%dm'] else None, subset = slice)

    #Dataframe ID and title 

    lq.set_caption("SEM Model Comparison ||  " + vbsem.cmeta['filename'] + ' vs. ref: || ' + refobj.cmeta['filename'] + '|| q = ' + str(qu) + '|| twidth =' + str(thresh['%dw']) + '|| tmean =' + str(thresh['%dm']) + '|| tlap =' + str(thresh['%lap']))

    #Dataframe, Print out details about threshold values

    return lq
# %%
def calc_goverlap(dsems, refobj = single_cfa_mcmc, qu = [0.025, 0.975]):
    '''
    Input
    dsems( dictionary): dictionary of dsems objects.

    Output:
    Columns: variables
    Row Names: dsems.filename
    '''
    data = {dsems[key].cmeta['filename']: calc_compmodels(dsems[key], refobj = refobj, qu = qu)['%vblap'] for key in dsems.keys()}

    q = pd.DataFrame.from_dict(data)

    return q.T
# %%
def calc_gcompmodel(dsems, refobj = single_cfa_mcmc, qu = [0.025, 0.975]):
    '''
    For overlap.
    Columns: variables
    Row Names: dsems.filename
    '''
    data = {dsems[key].cmeta['filename']: calc_compmodels(dsems[key], refobj = refobj, qu = qu)['%lap'] for key in dsems.keys()}

    q = pd.DataFrame.from_dict(data)

    return q.T

# %%
#Functions for the User: 
#global_overlap 
#global_compmeans
#global_compwidth

def global_overlap(dsems, refobj = single_cfa_mcmc, qu = [0.025, 0.975], t = 0.90, index_labels = None, index_name = None, title = None, filename = None):
    q = calc_goverlap(dsems, refobj = refobj, qu = qu)
    raw = '.png'
    if index_labels is not None:
        q.index = index_labels
        q.index.name = index_name
    else: 
        raw = 'raw.png'

    gq = q.style
    gq = gq.applymap(lambda v: 'color: red;' if v==0 else None).applymap(lambda v: 'color: green;' if (v>t) and (v< 1) else None).applymap(lambda v: 'color: orange;' if v==1 else None)

    gq.set_caption("CI Overlap as % of VB Interval ||  " + ' vs. ref: || ' + refobj.cmeta['filename'] + '|| q = ' + str(qu) + '|| threshold = ' + str(t))

    if title is not None:
        gq.set_caption(title)
   
    if filename is not None:
        gq.export_png(filename + raw)
    return gq
# %%
def global_compmeans(dsems, refobj = single_cfa_mcmc, t = 0.05, index_labels = None, index_name = None, title = None, filename = None, abs = False):

    raw = '.png'

    if abs:
        column_name = 'absmean'
    else:
        column_name = '%dmean'

    data = {dsems[key].cmeta['filename']: calc_compmodels(dsems[key], refobj = refobj)[column_name] for key in dsems.keys()}

    q = pd.DataFrame.from_dict(data).T

    if index_labels is not None:
        q.index = index_labels
        q.index.name = index_name
    else: 
        raw = 'raw.png'

    gq = q.style
    gq = gq.applymap(lambda v: 'opacity: 20%;' if (v < t) and (v > -t) else None).applymap(lambda v: 'color: blue;' if (v>t)else None).applymap(lambda v: 'color: black;' if v <-t else None)

    # if title is not None:
    #     gq.set_caption(title)
    # else:
    #     gq.set_caption(column_name + ' vs. ref: || ' + refobj.cmeta['filename'] + '|| threshold = ' + str(t))

    #export to png file
    if filename is not None:
       gq.export_png(filename + raw)
    return gq

# %%
def global_compwidth(dsems, refobj = single_cfa_mcmc, qu = [0.025, 0.975], t = 0.10, index_labels = None, index_name = None, title = None, filename = None, std = False):

    raw = '.png'

    if std:
        column_name = 'dstd'
    else:
        column_name = '%dwidth'

    data = {dsems[key].cmeta['filename']: calc_compmodels(dsems[key], refobj = refobj, qu = qu)[column_name] for key in dsems.keys()}

    q = pd.DataFrame.from_dict(data).T

    if index_labels is not None:
        q.index = index_labels
        q.index.name = index_name
    else:
        raw = 'raw.png'

    gq = q.style
    gq = gq.applymap(lambda v: 'opacity: 20%;' if (v < t) and (v > -t) else None).applymap(lambda v: 'color: blue;' if (v>t)else None).applymap(lambda v: 'color: black;' if v <-t else None)

    # if title is not None:
    #     gq.set_caption(title)

    # else:
    #  gq.set_caption(column_name + ' vs. ref: || ' + refobj.cmeta['filename'] + '|| q = ' + str(qu) +  '|| threshold = ' + str(t))
    
    if filename is not None:
        gq.export_png(filename+raw)
    return gq
# %%
def global_compmode(dsems, refobj = single_cfa_mcmc, t = 0.05):

    data = {dsems[key].cmeta['filename']: calc_compmodels(dsems[key], refobj = refobj)['%dmode'] for key in dsems.keys()}

    q = pd.DataFrame.from_dict(data).T
    gq = q.style
    gq = gq.applymap(lambda v: 'opacity: 20%;' if (v < t) and (v > -t) else None).applymap(lambda v: 'color: blue;' if (v>t)else None).applymap(lambda v: 'color: black;' if v <-t else None)

    gq.set_caption("%Diff Mode ||  " + ' vs. ref: || ' + refobj.cmeta['filename'] + '|| threshold = ' + str(t))   
    return gq
# %%
def global_compall(dsems, refobjc = single_cfa_mcmc,index_name = None, suffix = None):
    index = dsems.keys()
    #filename = filepath + filename

    global_compmeans(dsems = dsems, index_labels= index, index_name = index_name, filename = filename + 'means', title = 'DiffMean: Variational Mean Comparison to MCMC' + suffix)

    global_compwidth(dsems = dsems, index_labels = index, index_name = index_name, filename = filename + 'width', title = 'DiffWidth: Credible Interval Width Comparison to MCMC' + suffix)

    global_overlap(dsems = dsems, index_labels = index, index_name = index_name, filename = filename + 'overlap', title = 'Overlap: Comparison of Credible Interval Overlap with MCMC'+ suffix)

# %%
# Moment Calculation - Obtain alpha hat and beta hat 

