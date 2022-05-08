
# MNIST HVAE SL5
# MNIST IWAE SL3
# MNIST LVAE SL5

# OMNIGLOT HVAE SL5
# OMNIGLOT IWAE SL4
# OMNIGLOT LVAE SL5

import os
import matplotlib.pyplot as plt
import pandas as pd

""" MNIST PLOT """
# Import csv files
df1 = pd.read_csv('main_log_MNIST_HVAE.csv')
df2 = pd.read_csv('main_log_MNIST_IWAE.csv')
df3 = pd.read_csv('main_log_MNIST_LVAE.csv')

# Plot val2_loss
plt.figure()
plt.plot(df1['val2_loss'],linewidth=0.5)
plt.plot(df2['val2_loss'],linewidth=0.5)
plt.plot(df3['val2_loss'],linewidth=0.5)

# Set limits
min_val2_loss = min(df1['val2_loss'][50:].min(),df2['val2_loss'][50:].min(),df3['val2_loss'][50:].min())
max_val2_loss = max(df1['val2_loss'][50:].max(),df2['val2_loss'][50:].max(),df3['val2_loss'][50:].max())

plt.ylim(min_val2_loss,max_val2_loss)
plt.xlim(50,500)
plt.legend(['HVAE (SL5)','IWAE (SL3)','LVAE (SL5)'], fontsize=16)
plt.title('MNIST test loss (k=20)', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Loss',fontsize=16)

plt.savefig('plot7_MNIST.png')

""" OMNIGLOT PLOT """
# Import csv files
df1 = pd.read_csv('main_log_OMNIGLOT_HVAE.csv')
df2 = pd.read_csv('main_log_OMNIGLOT_IWAE.csv')
df3 = pd.read_csv('main_log_OMNIGLOT_LVAE.csv')

# Plot val2_loss
plt.figure()
plt.plot(df1['val2_loss'],linewidth=0.5)
plt.plot(df2['val2_loss'],linewidth=0.5)
plt.plot(df3['val2_loss'],linewidth=0.5)

# Set limits
min_val2_loss = min(df1['val2_loss'][50:].min(),df2['val2_loss'][50:].min(),df3['val2_loss'][50:].min())
max_val2_loss = max(df1['val2_loss'][50:].max(),df2['val2_loss'][50:].max(),df3['val2_loss'][50:].max())

plt.ylim(min_val2_loss,max_val2_loss)
plt.xlim(50,500)
plt.legend(['HVAE (SL5)','IWAE (SL4)','LVAE (SL5)'], fontsize=16)
plt.title('OMNIGLOT test loss (k=20)', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.savefig('plot7_OMNIGLOT.png')