import numpy as np
from math import ceil,log,log10,sqrt
import scipy.stats as st
import scipy.optimize as opt
import pandas as pd
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

# Gets current working directory
cwd = os.getcwd()
st = '/Users/erics/UCSD/DSC180A/DSC180A-Project-1/output1.csv'
# Inserts file path of algorithms.py and imports models
sys.path.insert(0, cwd+ '/src')
import algorithms as a

# Arguments from command prompt
args = (sys.argv)
fname = args[1]
df = pd.DataFrame([fname,os.getcwd()])
#df.to_csv(st,index=False)


mu2 = [0.01*x for x in range(0,101)]
df = a.simulationN_moss(mu2,n=1,num_sim=10,gaussian=True)


df.to_csv(os.getcwd()+'output.csv',index=False)
