
import os
import matplotlib.pyplot as plt
import pandas as pd

# Import csv files and make figure
df1 = pd.read_csv('main_log_ld8.csv')
df2 = pd.read_csv('main_log_ld16.csv')
df3 = pd.read_csv('main_log_ld32.csv')
df4 = pd.read_csv('main_log_ld64.csv')

fig, axs = plt.subplots(1,4, figsize=(22,5))

# Plot train loss
axs[0].plot(df1['train_loss'],linewidth=0.5)
axs[0].plot(df2['train_loss'],linewidth=0.5)
axs[0].plot(df3['train_loss'],linewidth=0.5)
axs[0].plot(df4['train_loss'],linewidth=0.5)

# Set limits
min_train_loss = min(df1['train_loss'].min(), df2['train_loss'].min(), df3['train_loss'].min(), df4['train_loss'].min())
max_train_loss = max(df1['train_loss'].max(), df2['train_loss'].max(), df3['train_loss'].max(), df4['train_loss'].max())

axs[0].set_ylim(min_train_loss, max_train_loss)
axs[0].set_xlim(0, 200)

# Set other stuff
axs[0].set_title('train loss', size=24)
axs[0].tick_params(axis='both', which='major', labelsize=18)
axs[0].set_ylabel('loss', size=20)
axs[0].set_xlabel('epochs', size=20)

# Plot val loss
axs[1].plot(df1['val_loss'],linewidth=0.5)
axs[1].plot(df2['val_loss'],linewidth=0.5)
axs[1].plot(df3['val_loss'],linewidth=0.5)
axs[1].plot(df4['val_loss'],linewidth=0.5)

# Set limits
min_test_loss = min(df1['val_loss'].min(), df2['val_loss'].min(), df3['val_loss'].min(), df4['val_loss'].min())
max_test_loss = max(df1['val_loss'].max(), df2['val_loss'].max(), df3['val_loss'].max(), df4['val_loss'].max())

axs[1].set_ylim(min_test_loss, max_test_loss)
axs[1].set_xlim(0, 200)

# Set other stuff
axs[1].set_title('test loss', size=24)
axs[1].tick_params(axis='both', which='major', labelsize=18)
axs[1].set_xlabel('epochs', size=20)

# Plot train rec_loss
axs[2].plot(df1['train_log_PxGz'],linewidth=0.5)
axs[2].plot(df2['train_log_PxGz'],linewidth=0.5)
axs[2].plot(df3['train_log_PxGz'],linewidth=0.5)
axs[2].plot(df4['train_log_PxGz'],linewidth=0.5)

# Set limits
min_rec_loss = min(df1['train_log_PxGz'].min(), df2['train_log_PxGz'].min(), df3['train_log_PxGz'].min(), df4['train_log_PxGz'].min())
max_rec_loss = max(df1['train_log_PxGz'].max(), df2['train_log_PxGz'].max(), df3['train_log_PxGz'].max(), df4['train_log_PxGz'].max())

axs[2].set_ylim(min_rec_loss, max_rec_loss)
axs[2].set_xlim(0, 200)

# Set other stuff
axs[2].set_title('train rec', size=24)
axs[2].tick_params(axis='both', which='major', labelsize=18)
axs[2].set_xlabel('epochs', size=20)

# Plot val rec_loss
axs[3].plot(df1['val_log_PxGz'],linewidth=0.5)
axs[3].plot(df2['val_log_PxGz'],linewidth=0.5)
axs[3].plot(df3['val_log_PxGz'],linewidth=0.5)
axs[3].plot(df4['val_log_PxGz'],linewidth=0.5)

# Set limits
min_rec_loss = min(df1['val_log_PxGz'].min(), df2['val_log_PxGz'].min(), df3['val_log_PxGz'].min(), df4['val_log_PxGz'].min())
max_rec_loss = max(df1['val_log_PxGz'].max(), df2['val_log_PxGz'].max(), df3['val_log_PxGz'].max(), df4['val_log_PxGz'].max())

axs[3].set_ylim(min_rec_loss, max_rec_loss)
axs[3].set_xlim(0, 200)

# Set other stuff
axs[3].set_title('test rec', size=24)
axs[3].tick_params(axis='both', which='major', labelsize=18)
axs[3].set_xlabel('epochs', size=20)
axs[3].legend(['LD8','LD16','LD32','LD64'], fontsize=18)

plt.subplots_adjust(bottom=0.15)

plt.tight_layout()
plt.savefig('DGD_plot2.png')



"""
# Shared legend and layout changes
axs[2].legend(['SL1','SL3','SL4','SL5'], fontsize=18)
plt.subplots_adjust(bottom=0.15)
plt.tight_layout()

fig.savefig('plot1.png')
plt.show()

print("LOL")

"""
