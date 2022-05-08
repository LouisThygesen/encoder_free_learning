
import os
import matplotlib.pyplot as plt
import pandas as pd

# Import csv file main_log_sl1.csv using pandas
df1 = pd.read_csv('main_log_sl1.csv')
df2 = pd.read_csv('main_log_sl2.csv')
df3 = pd.read_csv('main_log_sl3.csv')
df4 = pd.read_csv('main_log_sl4.csv')
df5 = pd.read_csv('main_log_sl5.csv')

fig, axs = plt.subplots(2,4, figsize=(22,10))

""" Part 1: Train loss"""
# Plot train loss
axs[0,0].plot(df1['train_loss'],linewidth=0.5)
axs[0,0].plot(df2['train_loss'],linewidth=0.5)
axs[0,0].plot(df3['train_loss'],linewidth=0.5)
axs[0,0].plot(df4['train_loss'],linewidth=0.5)
axs[0,0].plot(df5['train_loss'],linewidth=0.5)

# Set limits
min_train_loss = min(df1['train_loss'][10:].min(), df2['train_loss'][10:].min(), df3['train_loss'][10:].min(), df4['train_loss'][10:].min(), df5['train_loss'][10:].min())
max_train_loss = max(df1['train_loss'][10:].max(), df2['train_loss'][10:].max(), df3['train_loss'][10:].max(), df4['train_loss'][10:].max(), df5['train_loss'][10:].max())

axs[0,0].set_ylim(min_train_loss, max_train_loss)
axs[0,0].set_xlim(10, 200)

# Set other stuff
axs[0,0].set_title('train loss', size=24)
axs[0,0].tick_params(axis='both', which='major', labelsize=18)
axs[0,0].set_ylabel('loss', size=20)
axs[0,0].set_xlabel('epochs', size=20)

# Plot train rec-loss
axs[0,1].plot(df1['train_log_PxGz'],linewidth=0.5)
axs[0,1].plot(df2['train_log_PxGz'],linewidth=0.5)
axs[0,1].plot(df3['train_log_PxGz'],linewidth=0.5)
axs[0,1].plot(df4['train_log_PxGz'],linewidth=0.5)
axs[0,1].plot(df5['train_log_PxGz'],linewidth=0.5)

# Set limits
min_train_log_PxGz = min(df1['train_log_PxGz'][30:].min(), df2['train_log_PxGz'][30:].min(), df3['train_log_PxGz'][30:].min(), df4['train_log_PxGz'][30:].min(), df5['train_log_PxGz'][30:].min())
max_train_log_PxGz = max(df1['train_log_PxGz'][30:].max(), df2['train_log_PxGz'][30:].max(), df3['train_log_PxGz'][30:].max(), df4['train_log_PxGz'][30:].max(), df5['train_log_PxGz'][30:].max())

axs[0,1].set_ylim(min_train_log_PxGz, max_train_log_PxGz)
axs[0,1].set_xlim(30, 200)

# Set other stuff
axs[0,1].set_title('train rec-loss', size=24)
axs[0,1].tick_params(axis='both', which='major', labelsize=18)
axs[0,1].set_xlabel('epochs', size=20)
axs[0,1].legend(['SL1','SL2','SL3','SL4','SL5'], fontsize=18)

# Plot z_L-loss
axs[0,2].plot(df1['train_log_PzL'],linewidth=0.5)
axs[0,2].plot(df2['train_log_PzL'],linewidth=0.5)
axs[0,2].plot(df3['train_log_PzL'],linewidth=0.5)
axs[0,2].plot(df4['train_log_PzL'],linewidth=0.5)
axs[0,2].plot(df5['train_log_PzL'],linewidth=0.5)

# Set limits
min_train_log_PzL = min(df1['train_log_PzL'].min(), df2['train_log_PzL'].min(), df3['train_log_PzL'].min(), df4['train_log_PzL'].min(), df5['train_log_PzL'].min())
max_train_log_PzL = max(df1['train_log_PzL'].max(), df2['train_log_PzL'].max(), df3['train_log_PzL'].max(), df4['train_log_PzL'].max(), df5['train_log_PzL'].max())

axs[0,2].set_ylim(-5, max_train_log_PzL)
axs[0,2].set_xlim(0, 200)

# Set other stuff
axs[0,2].set_title('train z_L-loss', size=24)
axs[0,2].tick_params(axis='both', which='major', labelsize=18)
axs[0,2].set_xlabel('epochs', size=20)

# Plot rest of loss
axs[0,3].plot(df1['train_log_rest'],linewidth=0.5)
axs[0,3].plot(df2['train_log_rest'],linewidth=0.5)
axs[0,3].plot(df3['train_log_rest'],linewidth=0.5)
axs[0,3].plot(df4['train_log_rest'],linewidth=0.5)
axs[0,3].plot(df5['train_log_rest'],linewidth=0.5)

# Set limits
min_train_log_rest = min(df1['train_log_rest'][10:].min(), df2['train_log_rest'][10:].min(), df3['train_log_rest'][10:].min(), df4['train_log_rest'][10:].min(), df5['train_log_rest'][10:].min())
max_train_log_rest = max(df1['train_log_rest'][10:].max(), df2['train_log_rest'][10:].max(), df3['train_log_rest'][10:].max(), df4['train_log_rest'][10:].max(), df5['train_log_rest'][10:].max())

axs[0,3].set_ylim(min_train_log_rest,max_train_log_rest)
axs[0,3].set_xlim(10, 200)

# Set other stuff
axs[0,3].set_title('train rest of loss', size=24)
axs[0,3].tick_params(axis='both', which='major', labelsize=18)
axs[0,3].set_xlabel('epochs', size=20)

""" Part 2: Test loss """
# Plot test loss
axs[1,0].plot(df1['val_loss'],linewidth=0.5)
axs[1,0].plot(df2['val_loss'],linewidth=0.5)
axs[1,0].plot(df3['val_loss'],linewidth=0.5)
axs[1,0].plot(df4['val_loss'],linewidth=0.5)
axs[1,0].plot(df5['val_loss'],linewidth=0.5)

# Set limits
min_val_loss = min(df1['val_loss'][10:].min(), df2['val_loss'][10:].min(), df3['val_loss'][10:].min(), df4['val_loss'][10:].min(), df5['val_loss'][10:].min())
max_val_loss = max(df1['val_loss'][10:].max(), df2['val_loss'][10:].max(), df3['val_loss'][10:].max(), df4['val_loss'][10:].max(), df5['val_loss'][10:].max())

axs[1,0].set_ylim(min_val_loss, max_val_loss)
axs[1,0].set_xlim(10, 200)

# Set other stuff
axs[1,0].set_title('test loss', size=24)
axs[1,0].tick_params(axis='both', which='major', labelsize=18)
axs[1,0].set_ylabel('loss', size=20)
axs[1,0].set_xlabel('epochs', size=20)

# Plot train rec-loss
axs[1,1].plot(df1['val_log_PxGz'],linewidth=0.5)
axs[1,1].plot(df2['val_log_PxGz'],linewidth=0.5)
axs[1,1].plot(df3['val_log_PxGz'],linewidth=0.5)
axs[1,1].plot(df4['val_log_PxGz'],linewidth=0.5)
axs[1,1].plot(df5['val_log_PxGz'],linewidth=0.5)

# Set limits
min_val_log_PxGz = min(df1['val_log_PxGz'][30:].min(), df2['val_log_PxGz'][30:].min(), df3['val_log_PxGz'][30:].min(), df4['val_log_PxGz'][30:].min(), df5['val_log_PxGz'][30:].min())
max_val_log_PxGz = max(df1['val_log_PxGz'][30:].max(), df2['val_log_PxGz'][30:].max(), df3['val_log_PxGz'][30:].max(), df4['val_log_PxGz'][30:].max(), df5['val_log_PxGz'][30:].max())

axs[1,1].set_ylim(min_val_log_PxGz, max_val_log_PxGz)
axs[1,1].set_xlim(30, 200)

# Set other stuff
axs[1,1].set_title('test rec-loss', size=24)
axs[1,1].tick_params(axis='both', which='major', labelsize=18)
axs[1,1].set_xlabel('epochs', size=20)

# Plot z_L-loss
axs[1,2].plot(df1['val_log_PzL'],linewidth=0.5)
axs[1,2].plot(df2['val_log_PzL'],linewidth=0.5)
axs[1,2].plot(df3['val_log_PzL'],linewidth=0.5)
axs[1,2].plot(df4['val_log_PzL'],linewidth=0.5)
axs[1,2].plot(df5['val_log_PzL'],linewidth=0.5)

# Set limits
min_val_log_PzL = min(df1['val_log_PzL'].min(), df2['val_log_PzL'].min(), df3['val_log_PzL'].min(), df4['val_log_PzL'].min(), df5['val_log_PzL'].min())
max_val_log_PzL = max(df1['val_log_PzL'].max(), df2['val_log_PzL'].max(), df3['val_log_PzL'].max(), df4['val_log_PzL'].max(), df5['val_log_PzL'].max())

axs[1,2].set_ylim(min_val_log_PzL-2, max_val_log_PzL+2)
axs[1,2].set_xlim(0, 200)

# Set other stuff
axs[1,2].set_title('test z_L-loss', size=24)
axs[1,2].tick_params(axis='both', which='major', labelsize=18)
axs[1,2].set_xlabel('epochs', size=20)

# Plot rest of loss
axs[1,3].plot(df1['val_log_rest'],linewidth=0.5)
axs[1,3].plot(df2['val_log_rest'],linewidth=0.5)
axs[1,3].plot(df3['val_log_rest'],linewidth=0.5)
axs[1,3].plot(df4['val_log_rest'],linewidth=0.5)
axs[1,3].plot(df5['val_log_rest'],linewidth=0.5)

# Set limits
min_val_log_rest = min(df1['val_log_rest'][10:].min(), df2['val_log_rest'][10:].min(), df3['val_log_rest'][10:].min(), df4['val_log_rest'][10:].min(), df5['val_log_rest'][10:].min())
max_val_log_rest = max(df1['val_log_rest'][10:].max(), df2['val_log_rest'][10:].max(), df3['val_log_rest'][10:].max(), df4['val_log_rest'][10:].max(), df5['val_log_rest'][10:].max())

axs[1,3].set_ylim(min_val_log_rest,max_val_log_rest+2)
axs[1,3].set_xlim(10, 200)

# Set other stuff
axs[1,3].set_title('test rest of loss', size=24)
axs[1,3].tick_params(axis='both', which='major', labelsize=18)
axs[1,3].set_xlabel('epochs', size=20)

plt.subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig('HDGD_plot2.png')

