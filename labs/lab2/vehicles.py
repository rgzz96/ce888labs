import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./vehicles.csv')

old = df['Current fleet'].tolist()
new = df['New Fleet'].tolist()

new = [x for x in new if x !='nan']


plt.scatter(np.arange(len(old)),old, c='b')
plt.scatter(np.arange(len(new)),new, c='r')
plt.legend(labels = ['Old Fleet', 'New Fleet'])
plt.title('Fleet Size')
plt.xlabel('Number')
plt.ylabel('size')
plt.savefig("Scatterplot.png", bbox_inches='tight')
plt.show()

plt.hist(old)
plt.savefig("HistogramNew.png", bbox_inches='tight')
plt.show()

plt.hist(df['New Fleet'].dropna())
plt.savefig("HistogramOld.png", bbox_inches='tight')
plt.show()


	
