# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:36:54 2019

@author: rg18217
"""
def p_value(sample1, sample2, reps, size):
	count = 0

	sample = np.concatenate((sample1, sample2))

	for i in range(reps):
		samp = np.random.choice(sample, sample.size * 2, replace = False)
		samp1 = samp[:sample.size]
		samp2 = samp[smaple.size:]

		t_perm = samp2.mean() -samp1.mean()

		if t_perm > size:
			count += 1

	return count/reps



def power(sample1, sample2, reps, size, alpha):
    count = 0
    
    for i in range(reps):
        samp1 = np.random.choice(sample1, sample1.size, Replace = True)
        samp2 = np.random.choice(sample2, sample2.size, Replace = True)
    	pvalue = p_value(samp1, samp2, reps, size)
    
    if pvalue < (1-alpha):
    	count +=1
     
    return count/teps

    
