# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:08:16 2020

@author: Manish Jathan
"""


import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sampleFile = open('merchants_merge_df','rb')
merge_df = pickle.load('merchants_merge_df')


