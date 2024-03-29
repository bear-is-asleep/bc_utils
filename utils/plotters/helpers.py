import matplotlib.pyplot as plt
from random import choice
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import date
import sys
import seaborn as sns
import matplotlib
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
from bc_utils.pmtutils import pic
import matplotlib.markers as mmarkers
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

#Date info
day = date.today().strftime("%Y_%m_%d")


def use_science_style():
  if 'science' in plt.style.available:
    plt.style.use(['science','no-latex'])
  else:
    os.system('pip install SciencePlots -q')
    plt.style.reload_library()
    plt.style.use(['science','sns'])

#Organization
def make_plot_dir():
    #day = date.today().strftime("%Y_%m_%d")
    day = date.today().strftime("%Y_%m_%d")
    isDir = os.path.isdir("Plots/Plots_"+day)
    if isDir == False:
        os.system("mkdir -p Plots_" +day)
        os.system("mv -n Plots_" +day+"/ Plots/")

def save_plot(fname,fig=None,ftype='.jpg',dpi=300,folder_name=f'Plots_{day}'):
    #print(f'Plots/{folder_name}')
    os.system(f'mkdir -p Plots/{folder_name}')
    if fig == None:
      plt.savefig(f'{fname}{ftype}',bbox_inches = "tight",dpi=dpi)
    else:
      fig.savefig(f'{fname}{ftype}',bbox_inches = "tight",dpi=dpi)
    os.system("mv " + fname + f"* Plots/{folder_name}/")

def plot_stuff():
  matplotlib.rcParams['xtick.labelsize'] = 13
  matplotlib.rcParams['ytick.labelsize'] = 13 #Set to same as axis size 
  matplotlib.rcParams['axes.labelsize'] = 'large'
  matplotlib.rcParams['legend.fontsize'] = 'large'
  matplotlib.rcParams['axes.titlesize'] = 'x-large'
  matplotlib.rcParams['figure.figsize'] = [9,7]
  make_plot_dir()
  use_science_style()

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

def get_bincenters(binedges):
  #Return bin centers from edges
  length = len(binedges) - 1
  bincenters = np.zeros(length) #Minus 1 for loss o557.0f 1 value
  for i in range(length):
    bincenters[i] = (binedges[i] + binedges[i+1])/2
  return bincenters

def convert_p_str(parameters):
  s = ''
  last_key = list(parameters)[-1]
  for key in parameters.keys():
    if last_key == key:
      s+= f"{key} = {parameters[key]}"
    else:
      s+= f"{key} = {parameters[key]}\n"
  return s

def mscatter(x,y,ax=None, m=None, **kw):
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc