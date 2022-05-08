
import os
import matplotlib.pyplot as plt
import pandas as pd

# Import csv files
df1 = pd.read_csv('main_log_MNIST_HVAE.csv')
df2 = pd.read_csv('main_log_MNIST_IWAE.csv')
df3 = pd.read_csv('main_log_MNIST_LVAE.csv')
df4 = pd.read_csv('main_log_MNIST_DGD.csv')
df5 = pd.read_csv('main_log_MNIST_HDGD.csv')

# Plot val-rec loss
plt.figure(figsize=(10,5))
#plt.plot(df1['val2_rec'], label='HVAE')
#plt.plot(df2['val2_rec'], label='IWAE')
#plt.plot(df3['val2_rec'], label='LVAE')
#plt.plot(df4['val_log_PxGz'], label='DGD')
plt.plot(df5['val_log_PxGz'], label='HDGD')
plt.plot(df5['val_loss'], label='loss')
plt.legend()



plt.show()

