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
out = os.path.join(cwd, 'output')
st = '/Users/erics/UCSD/DSC180A/DSC180A-Project-1/output1.csv'
# Inserts file path of algorithms.py and imports models
sys.path.insert(0, cwd + '/src')
import algorithms as a

# Arguments from command prompt
args = (sys.argv)
fname = args[1]
m = [25,50,75,100,1000]
mu2 = [0.01*x for x in range(0,101)]
theta = []
if fname == 'test':
    out = os.path.join(cwd, 'test')
    #etc
    df1 = a.simulationN_ETC(mu2,m,n=10,num_sim=10,gaussian=1)
    df2 = a.simulationN_ETC(mu2,m,n=10,num_sim=10,gaussian=2)
    #ubc
    df3 = a.simulationN_standard(mu2,n=10,num_sim=10,gaussian=True)
    df4 = a.simulationN_asymptotic(mu2,n=10,num_sim=10,gaussian=True)
    df5 = a.simulationN_moss(mu2,n=10,num_sim=10,gaussian=True)
    df6 = a.simulationN_KL(mu2,n=10,num_sim=10,gaussian=True)
    #thompson
    p1 = [0,1]
    p2 = [0,1]
    df7 = a.simulationN_TS1(mu2,p1,p2,n=10,num_sim=10)
    p1 = [1,1]
    p2 = [1,1]
    df8 = a.simulation_TS2(mu2,p1,p2,n=10,num_sim=10)
    #lin bandit
    a_param = [0.1,-0.1]
    df9 = a.simulationN_LinUCB(a_param,theta,n=10,num_sim=10)
    df10 = a.simulationN_TSLin(a_param,theta,n=10,num_sim=10)
    #BOP
    p1 = [1,1]
    df11 = a.simulationN_BOP(mu2,p1,n=10,num_sim=10)

else:
    #etc
    df1 = a.simulationN_ETC(mu2,m,n=1000,num_sim=1000,gaussian=1)
    df2 = a.simulationN_ETC(mu2,m,n=1000,num_sim=1000,gaussian=2)
    #ubc
    df3 = a.simulationN_standard(mu2)
    df4 = a.simulationN_asymptotic(mu2)
    df5 = a.simulationN_moss(mu2)
    df6 = a.simulationN_KL(mu2)
    #thompson
    p1 = [0,1]
    p2 = [0,1]
    df7 = a.simulationN_TS1(mu2,p1,p2)
    p1 = [1,1]
    p2 = [1,1]
    df8 = a.simulation_TS2(mu2,p1,p2)
    #lin bandit
    a_param = [0.1,-0.1]
    df9 = a.simulationN_LinUCB(a_param,theta)
    df10 = a.simulationN_TSLin(a_param,theta)
    #BOP
    p1 = [1,1]
    df11 = a.simulationN_BOP(mu2,p1)

df1.to_csv(os.path.join(out, 'output1.csv'),index=False)
df2.to_csv(os.path.join(out, 'output2.csv'),index=False)
df3.to_csv(os.path.join(out, 'output3.csv'),index=False)
df4.to_csv(os.path.join(out, 'output4.csv'),index=False)
df5.to_csv(os.path.join(out, 'output5.csv'),index=False)
df6.to_csv(os.path.join(out, 'output6.csv'),index=False)
df7.to_csv(os.path.join(out, 'output7.csv'),index=False)
df8.to_csv(os.path.join(out, 'output8.csv'),index=False)
df9.to_csv(os.path.join(out, 'output9.csv'),index=False)
df10.to_csv(os.path.join(out, 'output10.csv'),index=False)
df11.to_csv(os.path.join(out, 'output11.csv'),index=False)




