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

#matplotlib.rcParams['axes.unicode_minus'] = False

xls = 16 #axis size
tls = 20 #title size
lls = 16 #legend size
tbs=10 #Text box size
#Small displacement for text
small = 5
matplotlib.rcParams['xtick.labelsize'] = 13
matplotlib.rcParams['ytick.labelsize'] = 13 #Set to same as axis size 
matplotlib.rcParams['axes.labelsize'] = 'large'
matplotlib.rcParams['legend.fontsize'] = 'large'
matplotlib.rcParams['axes.titlesize'] = 'x-large'
matplotlib.rcParams['figure.figsize'] = [9,7]
plt.rcParams['figure.figsize'] = (9,7)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

def use_science_style():
  if 'science' in plt.style.available:
    plt.style.use(['science','no-latex'])
  else:
    os.system('pip install SciencePlots -q')
    plt.style.reload_library()
    plt.style.use(['science','sns'])

#Organization
def make_plot_dir():
    day = date.today().strftime("%d_%m_%Y")
    isDir = os.path.isdir("Plots/Plots_"+day)
    if isDir == False:
        os.system("mkdir -p Plots_" +day)
        os.system("mv -n Plots_" +day+"/ Plots/")

def save_plot(fname):
    day = date.today().strftime("%d_%m_%Y")
    plt.savefig(f'{fname}.jpg',bbox_inches = "tight")
    os.system("mv " + fname + "* Plots/Plots_" +day+"/")

def plot_stuff():
  plt.rcParams['figure.figsize'] = (9,7)
  matplotlib.rcParams['xtick.labelsize'] = 13
  matplotlib.rcParams['ytick.labelsize'] = 13 #Set to same as axis size 
  matplotlib.rcParams['axes.labelsize'] = 'large'
  matplotlib.rcParams['legend.fontsize'] = 'large'
  matplotlib.rcParams['axes.titlesize'] = 'x-large'
  matplotlib.rcParams['figure.figsize'] = [9,7]
  make_plot_dir()
  use_science_style()
plot_stuff()

def remove_outliers(arr,max_dev=4):
  #Remove outliers based on standard deviations of data to keep
  median,std = np.median(arr),np.std(arr)
  zero_based = abs(arr-median)
  return arr[zero_based < max_dev * std]

def max_bin_height(ax,bins):
  #Get max bin heigth
  max_bin = 0
  for bar, b0, b1 in zip(ax.containers[0], bins[:-1], bins[1:]):
    if bar.get_height() > max_bin:
      max_bin = bar.get_height()
  return max_bin

def convert_p_str(parameters):
  s = ''
  last_key = list(parameters)[-1]
  for key in parameters.keys():
    if last_key == key:
      s+= f"{key} = {parameters[key][0]} {parameters[key][1]}"
    else:
      s+= f"{key} = {parameters[key][0]} {parameters[key][1]}\n"
  return s

def scatter_3d(df,x_key,y_key,c_key,pdg_key='pdg',pdg=11,title='',xlabel='',ylabel=''):
  #3D scatterplot, with colors as third axis
  #Filter df by pdg type
  df = df[df.loc[:,pdg_key] == pdg]
  if title == '':
    title = c_key
  if xlabel == '':
    xlabel = x_key
  if ylabel == '':
    ylabel = y_key
  x = df.loc[:,x_key]
  y = df.loc[:,y_key]
  c = df.loc[:,c_key]
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.scatter(x,y,c=c)
  ax.set_xlabel(xlabel,fontsize=xls)
  ax.set_ylabel(ylabel,fontsize=xls)
  ax.set_title(title,fontsize=tls)

def hist_scattype(df,hist_key,bw=-1,pdg_key='pdg',pdg=11,scat_key='scat_type',title='',xlabel='',ylabel=''):
  df = df[df.loc[:,pdg_key] == pdg]
  if title == '':
    title = hist_key
  if xlabel == '':
    xlabel = hist_key
  if ylabel == '':
    ylabel = 'Count'
  #Arrays to fill with data
  nu_mu = []
  nu_e = []
  nubar_mu =[]
  nubar_e = []

  #Fill arrays with data according to scat key
  #0: nu_mu
  #1: nu_e
  #2: nubar_mu
  #3: nubar_e
  for line,row in df.iterrows():
    if int(row.loc[scat_key]) == 0:
      nu_mu.append(row.loc[hist_key])
    if int(row.loc[scat_key]) == 1:
      nu_e.append(row.loc[hist_key])
    if int(row.loc[scat_key]) == 2:
      nubar_mu.append(row.loc[hist_key])
    if int(row.loc[scat_key]) == 3:
      nubar_e.append(row.loc[hist_key])
  if bw == -1: #Automatically set binwidth
    bins=np.linspace(min(nu_mu),max(nu_mu),15)
  else:
    bins=range(min(nu_mu), max(nu_mu) + bw,bw)

  fig = plt.figure()
  ax = fig.add_subplot()
  ax.hist(nu_mu,bins=bins,label=r'$\nu_\mu + e^-$')
  ax.hist(nu_e,bins=bins,label=r'$\nu_e + e^-$')
  ax.hist(nubar_mu,bins=bins,label=r'$\bar{\nu}_\mu + e^-$')
  ax.hist(nubar_e,bins=bins,label=r'$\bar{\nu}_e + e^-$')
  ax.set_xlabel(xlabel,fontsize=xls)
  ax.set_ylabel(ylabel,fontsize=xls)
  ax.set_title(title,fontsize=tls)
  ax.legend(fontsize=lls)
  return ax,fig

xls = tls = lls =18
def hist_scatback(scat,back,hist_key,sb=None,nbins=None,pdg_key='genie_primaries_pdg',pdg=11,
  scat_key='scat_type',back_key='background_type',title='',xlabel='',ylabel='',alpha=0.9,scale='linear',
  bw=None,stacked=False,histtype='bar',pdgs=[11],include_background=True,status_code=1,
  colors=['red','#7FFF00','yellow','cyan','blue'],edgecolor=None):
  #Histogram with background and scattered events overlayed, with each type of scattering and background
  scat_dfs = []
  back_dfs = []

  #Eliminate initial particles to avoid double coutning
  scat = scat[scat.loc[:,'genie_status_code'] == status_code]
  back = back[back.loc[:,'genie_status_code'] == status_code]
  for pdg in pdgs:
    scat_dfs.append(scat[abs(scat.loc[:,pdg_key]) == pdg])
    back_dfs.append(back[abs(back.loc[:,pdg_key]) == pdg])
  scat = pd.concat(scat_dfs)
  back = pd.concat(back_dfs)
  
  #scat = scat[abs(scat.loc[:,pdg_key]) == pdg]
  #back = back[abs(back.loc[:,pdg_key]) == pdg]
  if title == '':
    title = hist_key
  if xlabel == '':
    xlabel = hist_key
  if ylabel == '':
    ylabel = 'Count'
  #Arrays to fill with data
  nu_mu = []
  nu_e = []
  nubar_mu =[]
  nubar_e = []

  #Fill arrays with data according to scat key
  #0: nu_mu
  #1: nu_e
  #2: nubar_mu
  #3: nubar_e
  for line,row in scat.iterrows():
    if int(row.loc[scat_key]) == 0:
      nu_mu.append(row.loc[hist_key])
    if int(row.loc[scat_key]) == 1:
      nu_e.append(row.loc[hist_key])
    if int(row.loc[scat_key]) == 2:
      nubar_mu.append(row.loc[hist_key])
    if int(row.loc[scat_key]) == 3:
      nubar_e.append(row.loc[hist_key])

  cc_1p0pi = [] #T < 20 MeV proton
  cc_0p0pi = [] #0 pion, 0 proton
  cc_1gam = [] #Pion decays into one photon
  #Fill arrays with data according to back key
  for line,row in back.iterrows():
    if int(row.loc[back_key]) == 0: #T < 20 MeV proton,pion
      cc_0p0pi.append(row.loc[hist_key])
    if int(row.loc[back_key]) == 1: #Maybe pion decay eventually
      cc_1p0pi.append(row.loc[hist_key])
  if sb is None and include_background:
    parameters = {#'Signal/Background':f'{sb:.2f}',
                  'Signal Events':f'{len(scat)}',
                  'Background Events':f'{len(back)}'
                  #''
    }
  elif include_background:
    parameters = {'Signal/Background':f'{sb:.2f}',
                  'Signal Events':f'{len(scat)}',
                  'Background Events':f'{len(back)}'
                  #''
    }
  else:
    parameters = {#'Signal/Background':f'{sb:.2f}',
                  'Signal Events':f'{len(scat)}',
                  #'Background Events':f'{len(back)}'
                  #''
    }

  if bw is None and nbins is not None:
    bins=np.linspace(min(min(nu_mu,default=np.nan),
                  min(nu_e,default=np.nan),
                  min(nubar_mu,default=np.nan),
                  min(nubar_e,default=np.nan),
                  min(cc_1p0pi,default=np.nan),
                  min(cc_0p0pi,default=np.nan)),
                  max(max(nu_mu,default=np.nan),
                  max(nu_e,default=np.nan),
                  max(nubar_mu,default=np.nan),
                  max(nubar_e,default=np.nan),
                  max(cc_1p0pi,default=np.nan),
                  max(cc_0p0pi,default=np.nan)),nbins)
    parameters['Bins'] = f'{nbins}'
  elif bw is not None and nbins is None:
    bins=np.arange(min(min(nu_mu,default=np.nan),
                  min(nu_e,default=np.nan),
                  min(nubar_mu,default=np.nan),
                  min(nubar_e,default=np.nan),
                  min(cc_1p0pi,default=np.nan),
                  min(cc_0p0pi,default=np.nan)),
                  max(max(nu_mu,default=np.nan),
                  max(nu_e,default=np.nan),
                  max(nubar_mu,default=np.nan),
                  max(nubar_e,default=np.nan),
                  max(cc_1p0pi,default=np.nan),
                  max(cc_0p0pi,default=np.nan))+bw,bw)
    parameters['Binwidth']=f'{bw}'
  #print(parameters)
  ptext = plotters.convert_p_str(parameters)
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)


  fig = plt.figure(figsize=(9,7))
  ax = fig.add_subplot()
  #ax.hist(cc_1p0pi,bins=bins,label=r'CC 1$e$1$p$0$\pi$ ($T_p$ < 20 MeV)',alpha=alpha)
  if not stacked: #Plot each without stacking
    if include_background:
      ax.hist(cc_0p0pi,bins=bins,label=r'CC 1$e$0$p$0$\pi$',alpha=alpha,
              histtype=histtype,c=colors[4],edgecolor=edgecolor)
    ax.hist(nu_mu,bins=bins,label=r'$\nu_\mu + e^-$',alpha=alpha,
            histtype=histtype,c=colors[0],edgecolor=edgecolor)
    ax.hist(nu_e,bins=bins,label=r'$\nu_e + e^-$',alpha=alpha,
            histtype=histtype,c=colors[1],edgecolor=edgecolor)
    ax.hist(nubar_mu,bins=bins,label=r'$\bar{\nu}_\mu + e^-$',alpha=alpha,
            histtype=histtype,c=colors[2],edgecolor=edgecolor)
    ax.hist(nubar_e,bins=bins,label=r'$\bar{\nu}_e + e^-$',alpha=alpha,
            histtype=histtype,c=colors[3],edgecolor=edgecolor)
  else:
    labels = [r'$\nu_\mu + e^-$',r'$\nu_e + e^-$',
              r'$\bar{\nu}_\mu + e^-$',r'$\bar{\nu}_e + e^-$']
    nus = [nu_mu,nu_e,nubar_mu,nubar_e]
    if include_background: #Add background events
      labels.extend([r'CC 1$e$0$p$0$\pi$'])
      nus.append(cc_0p0pi)
    ax.hist(nus[::-1],bins=bins,stacked=stacked,histtype=histtype,label=labels[::-1],
    alpha=alpha,color=colors[::-1],edgecolor=edgecolor)#[::-1] reverses list

  ax.set_xlabel(xlabel,fontsize=xls)
  ax.set_ylabel(ylabel,fontsize=xls)
  ax.set_title(title,fontsize=tls)
  ax.legend(fontsize=lls)
  ax.set_yscale(scale)
  # place a text box in upper right in axes coords
  ax.text(0.6, 0.5, ptext, transform=ax.transAxes, fontsize=tbs+3,
        verticalalignment='top', bbox=props)
  return ax,fig





