import numpy as np
import numpy as np
import pandas as pd
from datetime import date
import sys
bc_utils_path = '/Users/bearcarlson/python_utils/'
sbnd_utils_path = '/sbnd/app/users/brindenc/mypython'
sys.path.append(sbnd_utils_path) #My utils path
from bc_utils.pmtutils import plotters as pmtplotters
from time import time
from bc_utils.utils import pic

def get_xy_bins(df,xkey,ykey,index,bw,module=-1,xmin=None,xmax=None):
  #df is input dataframe
  #xkey is x axis key to make bin sizes
  #ykey is y axis to make scale
  #index is which event we want
  #bw is bin wdith
  xs = df.loc[index,xkey].sort_index().values
  ys = df.loc[index,ykey].sort_index().values
  modules = df.loc[:,'crt_module'].drop_duplicates().values
  if xmax == None and xmin == None:
    binedges = np.arange(xs.min(),xs.max()+bw,bw)
    trim_edges = False
  else:
    binedges = np.arange(xmin,xmax+bw,bw)
    trim_edges = True #Digitize keeps excess data in extra bins, first and last ones. We should drop these since they're out of the region of interest
  if module == -1:
    check_all = True #Check all adc at once
  else:
    check_all = False
  

  #Make title for plots
  y_hist = np.zeros(len(binedges)+1) #Histogram values, add one buffer
  inds = np.digitize(xs,binedges) #Returns list with indeces where value belongs
  for i,ind in enumerate(inds):
    if check_all:
      y_hist[ind] += ys[i] #Add y-val for any crt
    elif check_all and modules[i] == module:
      y_hist[ind] += ys[i] #Add y-val for any crt module

  #bincenters = binedges - bw/2 #temp fix of x-axis not being centered
  bincenters = binedges #temp fix of x-axis not being centered
  if trim_edges:
    #Delete first and last elements
    bincenters = np.delete(bincenters,0)
    bincenters = np.delete(bincenters,-1)
    y_hist = np.delete(y_hist,0)
    y_hist = np.delete(y_hist,-1)
  return bincenters,y_hist

