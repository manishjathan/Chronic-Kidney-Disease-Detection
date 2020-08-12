# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:58:27 2020

@author: capiot
"""

import numpy as np
import pandas as pd
import pickle

median_sod = 138.0

#scaler = pickle.load(open('scaler', 'rb'))


## Let's write a function to perform all the required transformations for the cross validation data
def formBins(x,lst):
  if x <= lst[0]:
    return 0
  elif x > lst[0] and x <= lst[1]: 
    return 1
  elif x > lst[1] and x <= lst[2]: 
    return 2
  elif x > lst[2]:
    return 3

def formAlBins(x):
  if x == 0:
    return 0
  elif x >=1 and x <=2:
    return 1
  else:
    return 2


def reflector(x):
  
  if x < median_sod:
    dev = median_sod-x
    return median_sod + dev
  elif x > median_sod:
    dev = x - median_sod
    return median_sod - dev
  else:
    return x

def sc_bu_bin(x):
    if x['sc'] <= 1.4 and x['bu'] <= 50:
        return 0
    else:
        return 1
    
def transformAttributes(features):
    
    """
    Columns to be generated
        'target', 'age_bins', 'bp_bins', 'al_cat', 'su_bin', 'bgr_bin',
       'bu_bin', 'sc_bin', 'log_norm_sc', 'sod_bin', 'log_norm_sod',
       'norm_sod_bin', 'hemo_bin', 'rc_bin', 'wc_bin', 'sc_bu_bin', 'acr',
       'multivariate_pdf', 'log_multivariate_pdf', 'log_multi_pdf_bin
    """
    attr_cols = ['age','bp','al','su','bgr','bu','sc','sod','hemo','rc','wc']
    ## Use reshape instead
    
    feature_array = np.array(features).reshape([-1,len(features)])
    test_df = pd.DataFrame(feature_array,columns = attr_cols)
    #print(test_df.head(1))
    
    trans_df = pd.DataFrame()
    trans_df['age_bins'] = test_df['age'].apply(lambda x : formBins(x,[20,40,60]))
    
    #trans_df['bp_bins'] = test_df['bp'].apply(lambda x : formBins(x,[60,80,90]))
    trans_df['al_cat'] = test_df['al'].apply(lambda x : formAlBins(x))
    trans_df['su_bin'] = test_df['su'].apply(lambda x : 0 if x==0 else 1)
    trans_df['bgr_bin'] = test_df['bgr'].apply(lambda x : 0 if x <= 140 else 1)
    trans_df['bu_bin'] = test_df['bu'].apply(lambda x : 0 if x <= 50 else 1)
    trans_df['sc_bin'] = test_df['sc'].apply(lambda x : 0 if x <= 1.2 else 1)
    trans_df['log_norm_sc'] = np.log(test_df['sc'])
    trans_df['sod_bin'] = test_df['sod'].apply(lambda x : 1 if x <= 138 else 0)
    
    reflected_sod = test_df['sod'].apply(lambda x : reflector(x))
    log_norm_sod = np.log(reflected_sod)
    trans_df['log_norm_sod'] = log_norm_sod
    
    trans_df['norm_sod_bin'] = trans_df['log_norm_sod'].apply(lambda x : 0 if x <= 4.92 else 1)
    trans_df['hemo_bin'] = test_df['hemo'].apply(lambda x : 0 if x <= 12.65 else 1)
    trans_df['rc_bin'] = test_df['rc'].apply(lambda x : 0 if x <= 4.4 else 1)
    trans_df['wc_bin'] = test_df['wc'].apply(lambda x : 0 if x <=8600 else 1)
    trans_df['sc_bu_bin'] = test_df.apply(lambda x : sc_bu_bin(x),axis = 1)
    
    trans_df['acr'] = test_df['al']/test_df['sc']
    
    x = trans_df
    return x
