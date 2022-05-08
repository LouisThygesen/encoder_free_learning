
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# DATA MNIST
y1 = np.array([88.82,float('nan'),88.01,88.50,87.53]) # MNIST HVAE
y2 = np.array([87.93,87.56,87.54,87.64,87.56]) # MNIST IWAE
y3 = np.array([float('nan'),86.27,86.10,86.01,86.01]) # MNIST LVAE
x = np.arange(1,6)

# plot 3 graphs (connected dots) - one for each dataset
fig = plt.figure()
ax = fig.gca()

ax.plot(np.array([1,3]),np.array([88.82,88.01]),'r')
ax.plot(x,y1,'ro-',label='HVAE')
ax.plot(x,y2,'go-',label='IWAE')
ax.plot(x,y3,'bo-',label='LVAE')

# Style
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(loc=3,fontsize=16)
plt.title('MNIST test loss (k=20)',fontsize=20)
plt.xlabel('Stochastic layers',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.savefig('plot8_MNIST.png')

# DATA OMNIGLOT
y4 = np.array([117.61,116.7,116.81,116.57,116.39]) # OMNIGLOT HVAE
y5 = np.array([112.89,112.96,112.75,112.58,112.67]) # OMNIGLOT IWAE
y6 = np.array([float('nan'),109.69,109.46,109.60,109.39]) # OMNIGLOT LVAE

plt.figure()
plt.plot(x,y4,'ro-',label='HVAE')
plt.plot(x,y5,'go-',label='IWAE')
plt.plot(x,y6,'bo-',label='LVAE')
plt.legend()

# Style
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(loc=3,fontsize=16)
plt.title('OMNIGLOT test loss (k=20)',fontsize=20)
plt.xlabel('Stochastic layers',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.savefig('plot8_OMNIGLOT.png')