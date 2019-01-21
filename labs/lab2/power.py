# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:36:54 2019

@author: rg18217
"""

def power(sample1, sample2, reps, size, alpha):
    count = 0
    observed = sample2.mean() - sample1.mean()
    
    for i in range(reps):
        s1 = np.random.choice(sample1, size = size, Replace = True)
        s2 = np.random.choice(sample1, size = size, Replace = True)
    
    perm = s1.mean() - s2.mean()
    
    if perm < 1-alpha:
        count+=1
    
    ratio = count/reps
    
    return ratio

    
