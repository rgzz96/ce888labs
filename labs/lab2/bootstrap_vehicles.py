import matplotlib
#matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np


def boostrap(sample, sample_size, iterations):
	# <---INSERT YOUR CODE HERE--->
   arr1 = np.empty([iterations, sample_size])
   pmean = np.empty([iterations, 1])
   
   for i in range(iterations):
       arr1[i] = np.random.choice(sample, size=sample_size)
       pmean[i] = np.mean(arr1[i])
       
   data_mean = np.mean(pmean)
   lower = np.percentile(pmean, 2.5)
   upper = np.percentile(pmean, 97.5)
   return data_mean, lower, upper


if __name__ == "__main__":
    df = pd.read_csv('./vehicles.csv')
    i = 10000
    data_current = df.values.T[0]
    boot = boostrap(data_current, data_current.shape[0], i)
    print("Current Fleet")
    print(boot)
    
    data_new = df.values.T[1]
    data_new = data_new[~np.isnan(data_new)]
    boot = boostrap(data_new, data_new.shape[0], i)
    print("New Fleet")
    print(boot)
   







	#print ("Mean: %f")%(np.mean(data))
	#print ("Var: %f")%(np.var(data))
	


	