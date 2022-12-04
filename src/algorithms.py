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


# UCB

def pullGaussian(mu):
    return np.random.normal(mu,1)
    
def pullBernoulli(p):
    return np.random.binomial(1,p)

def simulationN_standard(mu2,n=1000,num_sim=1000,gaussian=True):
    point_lst = []
    var_lst = []
    df = pd.DataFrame()
    for m in tqdm(mu2):
        simulation = [UCB_standard(n=n,mu2=m,gaussian=1) for a in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df['point'] = point_lst
    df['var'] = var_lst
    return df

def simulationN_asymptotic(mu2,n=1000,num_sim=1000,gaussian=True):
    point_lst = []
    var_lst = []
    df = pd.DataFrame()
    for m in tqdm(mu2):
        simulation = [UCB_asymptotic(n=n,mu2=m,gaussian=1) for a in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df['point'] = point_lst
    df['var'] = var_lst
    return df

def simulationN_moss(mu2,n=1000,num_sim=1000,gaussian=True):
    point_lst = []
    var_lst = []
    df = pd.DataFrame()
    for m in tqdm(mu2):
        simulation = [UCB_moss(n=n,mu2=m,gaussian=1) for a in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df['point'] = point_lst
    df['var'] = var_lst
    return df

def simulationN_KL(mu2,n=1000,num_sim=1000,gaussian=True):
    point_lst = []
    var_lst = []
    df = pd.DataFrame()
    for m in tqdm(mu2):
        simulation = [UCB_KL(n=n,mu2=m,gaussian=1) for a in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df['point'] = point_lst
    df['var'] = var_lst
    return df

def UCB_standard(n,mu2,gaussian=1):
    if mu2 == 0:
        return 0
    reward = [pullGaussian(0),pullGaussian(mu2)] if gaussian else [pullBernoulli(0),pullBernoulli(mu2)]
    rewards1 = [reward[0]]
    rewards2 = [reward[1]]
    ti = [1,1]
    t = 2
    regret = 0
    optimal = 0 if mu2 < 0 else 1
    
    while(t < n):
        ucb = [np.mean(rewards1) + np.sqrt(2*log(n**2)/ti[0]),np.mean(rewards2) + np.sqrt(2*log(n**2)/ti[1])]
        argmax = np.argmax(ucb)
        reward = [pullGaussian(0),pullGaussian(mu2)] if gaussian else [pullBernoulli(0),pullBernoulli(mu2)]
        
        if argmax == 0:
            rewards1 += [reward[0]]
        else:
            rewards2 += [reward[1]]
        ti[argmax] += 1
        t+=1
        regret += 0 if optimal == argmax else mu2
    
    return regret

def UCB_asymptotic(n,mu2,gaussian=1):
    if mu2 == 0:
        return 0
    reward = [pullGaussian(0),pullGaussian(mu2)] if gaussian else [pullBernoulli(0),pullBernoulli(mu2)]
    rewards1 = [reward[0]]
    rewards2 = [reward[1]]
    ti = [1,1]
    t = 2
    regret = 0
    optimal = 0 if mu2 < 0 else 1
    
    while(t < n):
        ft = np.log(1 + t*np.log(np.log(t)))
        ucb = [np.mean(rewards1) + np.sqrt(2*ft/ti[0]),np.mean(rewards2) + np.sqrt(2*ft/ti[1])]
        argmax = np.argmax(ucb)
        reward = [pullGaussian(0),pullGaussian(mu2)] if gaussian else [pullBernoulli(0),pullBernoulli(mu2)]
        
        if argmax == 0:
            rewards1 += [reward[0]]
        else:
            rewards2 += [reward[1]]
        ti[argmax] += 1
        t+=1
        regret += 0 if optimal == argmax else mu2
    
    return regret

def log_plus(n,t):
    x = n / (2  *t)
    return max(np.log(1),np.log(x))

def UCB_moss(n,mu2,gaussian=1):
    if mu2 == 0: 
        return 0
    reward = [pullGaussian(0),pullGaussian(mu2)] if gaussian else [pullBernoulli(0),pullBernoulli(mu2)]
    rewards1 = [reward[0]]
    rewards2 = [reward[1]]
    ti = [1,1]
    t = 2
    regret = 0
    optimal = 0 if mu2 < 0 else 1
    
    while(t < n):
        if t != 1:
            ft = 1 + t*log(t)*log(t)
            
        ucb = [np.mean(rewards1) + np.sqrt((4/ti[0])*log_plus(n,ti[0])),np.mean(rewards2) + np.sqrt((4/ti[1])*log_plus(n,ti[1]))]
        argmax = np.argmax(ucb)
        reward = [pullGaussian(0),pullGaussian(mu2)] if gaussian else [pullBernoulli(0),pullBernoulli(mu2)]
        
        if argmax == 0:
            rewards1 += [reward[0]]
        else:
            rewards2 += [reward[1]]
        ti[argmax] += 1
        t+=1
        regret += 0 if optimal == argmax else mu2
    
    return regret

def d(p,q):
    if (p == 0):
        if (q < 1 and q > 0):
            return log(1/(1-q))
        else:
            return 0
    if (p == 1):
        if (q < 1 and q > 0):
            return log(1/q)
        else:
            return 1
    return p*log(p/q) + (1-p)*log((1-p)/(1-q))

def calculate_ucb(p,t,ti):
    ft = 1 + t*(log(log(t)))
    upper_bound = log(ft) / ti
    #bounds = [0,1]
    l = p
    r = np.array([1,1])
    for i in range(10):
        q = (l + r) / 2
        ndx = (np.where(p > 0, p * np.log(p / q), 0) +
             np.where(p < 1, (1 - p) * np.log((1 - p) / (1 - q)), 0)) < upper_bound
        l[ndx] = q[ndx]
        r[~ndx] = q[~ndx]
        #half = (sum(bounds)) / 2
        #if bounds[1]-bounds[0] < 1e-5:
            # early stopping
        #    break
        
        
        #entropy = d(p,half)
        
        #if (d(p,half) + d)
        #    bounds[0] = d(p,half) if p > 0 else 0
        #    bounds[1] = d(1-p,1-half) if p < 1 else 0

        #if entropy < upper_bound:
        #    bounds[0] = half
        #@else:
        #    bounds[1] = half
    #print(p)
    #print(half)
    return q

def UCB_KL(n,mu2,gaussian=1):
    if mu2 == 0.5:
        return 0
    reward = [pullBernoulli(0.5),pullBernoulli(mu2)]
    rewards1 = [reward[0]]
    rewards2 = [reward[1]]
    ti = [1,1]
    t = 2
    regret = 0
    optimal = 0 if mu2 < 0.5 else 1
    
    while(t < n): 
        ucb = calculate_ucb(np.array([np.mean(rewards1),np.mean(rewards2)]),t,ti[0])
        #print(ucb)
        argmax = np.argmax(ucb)
        
        reward = [pullBernoulli(0.5),pullBernoulli(mu2)]
        
        if argmax == 0:
            rewards1 += [reward[0]]
        else:
            rewards2 += [reward[1]]
        ti[argmax] += 1
        t+=1
        regret += 0 if optimal == argmax else abs(mu2-0.5)
    
    return regret