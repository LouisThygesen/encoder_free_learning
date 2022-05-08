
import os
import matplotlib.pyplot as plt
import pandas as pd

# Import csv file main_log_sl1.csv using pandas
df1 = pd.read_csv('main_log_sl1.csv')
df3 = pd.read_csv('main_log_sl3.csv')
df4 = pd.read_csv('main_log_sl4.csv')
df5 = pd.read_csv('main_log_sl5.csv')

# Plot train_loss
fig, axs = plt.subplots(1,3, figsize=(22,5))
axs[0].plot(df1['train_loss'],linewidth=0.5)
axs[0].plot(df3['train_loss'],linewidth=0.5)
axs[0].plot(df4['train_loss'],linewidth=0.5)
axs[0].plot(df5['train_loss'],linewidth=0.5)

# Set limits
min_train_loss = min(df1['train_loss'][50:].min(),df3['train_loss'][50:].min(),df4['train_loss'][50:].min(),df5['train_loss'][50:].min())
max_train_loss = max(df1['train_loss'][50:].max(),df3['train_loss'][50:].max(),df4['train_loss'][50:].max(),df5['train_loss'][50:].max())

axs[0].set_ylim(min_train_loss,97)
axs[0].set_xlim(50,500)

# Set other stuff
axs[0].set_title('train loss (k = 1)', size=24)
axs[0].tick_params(axis='both', which='major', labelsize=18)
axs[0].set_ylabel('loss', size=20)

# Plot val1_loss
axs[1].plot(df1['val1_loss'],linewidth=0.5)
axs[1].plot(df3['val1_loss'],linewidth=0.5)
axs[1].plot(df4['val1_loss'],linewidth=0.5)
axs[1].plot(df5['val1_loss'],linewidth=0.5)

# Set limits
min_val1_loss = min(df1['val1_loss'][50:].min(),df3['val1_loss'][50:].min(),df4['val1_loss'][50:].min(),df5['val1_loss'][50:].min())
max_val1_loss = max(df1['val1_loss'][50:].max(),df3['val1_loss'][50:].max(),df4['val1_loss'][50:].max(),df5['val1_loss'][50:].max())

axs[1].set_ylim(91.3,97)
axs[1].set_xlim(50,500)

# Set other stuff
axs[1].set_title('test loss (k = 1)', size=24)
axs[1].tick_params(axis='both', which='major', labelsize=18)
axs[1].set_xlabel('epoch', size=20)

# Plot val2_loss
axs[2].plot(df1['val2_loss'],linewidth=0.5)
axs[2].plot(df3['val2_loss'],linewidth=0.5)
axs[2].plot(df4['val2_loss'],linewidth=0.5)
axs[2].plot(df5['val2_loss'],linewidth=0.5)

# Define limits
min_val2_loss = min(df1['val2_loss'][50:].min(),df3['val2_loss'][50:].min(),df4['val2_loss'][50:].min(),df5['val2_loss'][50:].min())
max_val2_loss = max(df1['val2_loss'][50:].max(),df3['val2_loss'][50:].max(),df4['val2_loss'][50:].max(),df5['val2_loss'][50:].max())

axs[2].set_ylim(87.5,93)
axs[2].set_xlim(50,500)

# Set other stuff
axs[2].set_title('test loss (k = 20)', size=24)
axs[2].tick_params(axis='both', which='major', labelsize=18)

# Shared legend and layout changes
axs[2].legend(['SL1','SL3','SL4','SL5'], fontsize=18)
plt.subplots_adjust(bottom=0.15)
plt.tight_layout()

fig.savefig('plot1.png')
plt.show()

print("LOL")


