import numpy as np
from math import ceil,log,log10,sqrt
import scipy.stats as st
import scipy.optimize as opt
import pandas as pd
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt


#  Stochastic Bandits 

# ETC
# a: the arm we choose to pull
# mu2: the true value of the mean for arm 2
def pullGaussian1(a,mu2):
        if a == 1:
            return np.random.normal(0,1)
        return np.random.normal(mu2,1)
    
def pullBernoulli1(a,p):
        if a == 1:
            p = 0.5
            return np.random.binomial(1,p)
        return np.random.binomial(1,p)

# m is the number of times we explore arm a
# n is the horizons or the number of times we play
# mu2 is the mean of arm bandit 2
def ETC(m,n,mu2,comment=False,gaussian=1):
    arm_means = [0,0]
    true_mean = [0,mu2]
    arm_pulls = [0,0]
    if gaussian == 1:
        mu1 = 0
        optimal = mu2 if mu2 > mu1 else mu1
    else:
        mu1 = 0.5
        optimal = mu2 if mu2 > mu1 else mu1
    
    # exploration phase
    exploration_regret = (optimal - mu2)*m + (optimal - mu1)*m
    
    if gaussian == 1:
        reward_1 = [pullGaussian1(1,mu2) for a in range(m)]
        reward_2 = [pullGaussian1(2,mu2) for a in range(m)]
    else:
        reward_1 = [pullBernoulli1(1,mu2) for a in range(m)]
        reward_2 = [pullBernoulli1(2,mu2) for a in range(m)]
    empirical_mean_1 = np.mean(reward_1)
    empirical_mean_2 = np.mean(reward_2)
    # exploitation phase
    best_mean = mu2 if empirical_mean_1 < empirical_mean_2 else mu1
    best_arm = 1 if empirical_mean_1 < empirical_mean_2 else 0
    if comment:
        print("arm1 mean:" + str(mu1))
        print("arm2 mean:" + str(mu2))
        print("best arm:" + str(best_arm))
        print("optimal arm:" + str(optimal))
    
    #reward_exploit = [pullGaussian(best_arm,mu2) for i in range(n - 2*m)]
    #reward = pullGaussian(best_arm,mu2)
    exploitation_regret = (optimal - best_mean)*(n-2*m)
    
        
    total_regret = exploitation_regret + exploration_regret
    
    if comment:
        print("exploration regret:" + str(exploration_regret))
        print("exploitation regret:" + str(exploitation_regret))
        print("total regret:" + str(total_regret))
        print("best arm true mean:" + str(true_mean[best_arm-1]))
        print("\n")
    
    
    return total_regret

# simulating ETC's N parameter
# Function takes in lists for m, n (horizon), and mu2 and
# performs a grid search of each set of parameters.
# Will ignore permutations where n <= m.
def simulationN_ETC(mu2,m,n=1000,num_sim=1000,gaussian=1):
    df = pd.DataFrame()
    df['mu2'] = mu2
    det = [determine_m(a) for a in mu2]
    for j in tqdm(m):            
        point_lst = []
        err_lst = []
        for i in tqdm(mu2):
            if j != 1000:
                simulation = [ETC(m=j,n=n,mu2=i,gaussian=gaussian) for a in range(num_sim)]
            else:
                simulation = [ETC(m=determine_m(i),n=n,mu2=i,gaussian=gaussian) for a in range(num_sim)]
            point = np.mean(simulation)
            err = np.var(simulation)
            point_lst += [point]
            err_lst += [err]
        
        df[str(j) + "point"] = point_lst
        df[str(j) + "error"] = err_lst
    return df

def determine_m(mu2):
    return int(max(1,np.ceil(4*np.log(250*mu2**2)/mu2**2)))