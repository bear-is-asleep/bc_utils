import matplotlib.pyplot as plt
from random import choice
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import date
import sys
import seaborn as sns
import matplotlib
from scipy import optimize
sys.path.append('/sbnd/app/users/brindenc/mypython') #My utils path
from bc_utils.utils import pic,plotters

#Constants
xls = 16 #axis size
tls = 20 #title size
lls = 16 #legend size

#Functions
def plot_background_cuts(cuts_dfs,legend_labels,columns,title,scale='linear',ax=None,
  **kwargs):
  """Plots cuts made and events at each stage, made in cuts script
  cuts_dfs: signal first, backgrounds next list of dataframes
  columns: which cuts you want to show"""
  #List of pd series to plot
  means = []
  errs = []
  #Get data
  labels = list(cuts_dfs[0].keys())
  for df in cuts_dfs:
    means.append(df.iloc[-1,columns])
    errs.append(df.iloc[-2,columns])
  
  #Make fig and ax
  fig = plt.figure(figsize=(18,4))
  if ax is None:
    ax = fig.add_subplot()
  for i,mean in enumerate(means):
    ax.errorbar(columns,mean,yerr=errs[i],label=legend_labels[i],linewidth=3,**kwargs)
    ax.fill_between(columns,mean-errs[i],mean+errs[i],alpha=0.2,**kwargs)
  plt.xticks(columns,[labels[i] for i in columns])
  plt.setp(ax.get_xticklabels(), fontsize=xls,rotation=40)
  plt.setp(ax.get_yticklabels(), fontsize=xls)
  ax.legend(fontsize=lls)
  ax.set_ylabel('Events',fontsize=xls+2)
  ax.set_title(title,fontsize=tls)
  ax.set_yscale(scale)
  ax.set_ylim([0,None])
  return ax

  
def plot_efficiency_purity(cuts_dfs,columns,title,ax=None,event=0,**kwargs):
  """Plots cuts made and events at each stage, made in cuts script
  cuts_dfs: signal first, backgrounds next list of dataframes
  columns: which cuts you want to show, if None it'll show all
  event: select which event to view"""
  #List of pd series to plot
  events = []
  labels = list(cuts_dfs[0].keys())
  #Get data
  for df in cuts_dfs:
    events.append(df.iloc[event,columns].values) #Append info from first sample
  fig = plt.figure(figsize=(18,4))
  if ax is None:
    ax = fig.add_subplot()
  purity = events[0]/(np.sum(events,axis=0))
  efficiency = events[0]/events[0][0]
  ax.plot(columns,purity,label='Purity',linewidth=3,**kwargs)
  ax.plot(columns,efficiency,label='Efficiency',linewidth=3,**kwargs)
  plt.xticks(columns,[labels[i] for i in columns])
  plt.setp(ax.get_xticklabels(), fontsize=xls,rotation=40)
  plt.setp(ax.get_yticklabels(), fontsize=xls)
  ax.set_ylabel('Purity/Efficiency',fontsize=xls+2)
  ax.legend(fontsize=lls)
  ax.set_title(title,fontsize=tls)
  return ax

def back_sig_hist(dfs,labels,key,precut_events,title=None,xlabel=None,ylabel='Count',
  annotate=True,**kwarg):
  """Plot background and signal for key value
  dfs: 1st df is signal, rest are backgrounds
  precut_events: list of signal event count before cuts"""
  label_counts = labels.copy() #Avoid modifying the labels list
  events = []
  xs = []
  for i,df in enumerate(dfs):
    count = len(df.index.drop_duplicates())
    events.append(count) #Find event counts
    xs.append(df.loc[:,key].values) #Make histogram bins for each background
    label_counts[i] = label_counts[i] + f' ({count:.0f})'
  #print(events,np.sum(events))
  
  fig = plt.figure(figsize=(9,7))
  ax = fig.add_subplot()
  (n, bins, patches) = ax.hist(xs,stacked=True,label=label_counts,**kwarg) 
  purity = events[0]/np.sum(events)
  efficiency = events[0]/precut_events
  params = {'Purity':f'{purity*100:.2f}%',
    'Efficiency':f'{efficiency*100:.2f}%'}
    #'Binwidth':f'{bins[1]-bins[0]}'}
  if annotate:
    ptext = plotters.convert_p_str(params) #Convert params to text form to plot
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.9)
    ax.text(0.6, 0.5, ptext, transform=ax.transAxes, fontsize=lls,
          verticalalignment='top', bbox=props)
    if xlabel is None:
      xlabel = key
    ax.set_xlabel(xlabel,fontsize=xls)
    ax.set_ylabel(ylabel,fontsize=xls)
    ax.legend(fontsize=lls)
    if title is not None:
      ax.set_title(title,fontsize=tls)
  return ax,bins
  









