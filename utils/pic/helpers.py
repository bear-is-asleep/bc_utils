import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import os
from datetime import date
import sys
from scipy import optimize
from sklearn.linear_model import LinearRegression
from numpy import isin, sqrt,exp,arctan,cos,sin,pi
from time import time
from random import randint

#Constants

#Gaussian fit
def gaussian(x,a, mean, stddev):
  #Ex: popt, pcov = optimize.curve_fit(gaussian, x, data)
  return a*np.exp(-((x - mean) / sqrt(2) / stddev)**2)

def fit_gaussian(data):
  #Fits gaussian to 1d data and returns parameters
  steps=len(data)
  x = np.linspace(min(data),max(data),steps)
  popt, pcov = optimize.curve_fit(gaussian, x, data)
  return x,popt,pcov
def fit_gauss(x,y):
  return optimize.curve_fit(gaussian,x,y,sigma=sqrt(y))

def sum_combine(arr,every_other=2,col_options=[]):
  """
  DESC: Somes int=every_other for every other row
  arr: 2D array
  every_other: How many rows to combine, i.e. every_other=2 combines every other 
               line
  col_options: Enter column options 1D array same number of columns as arr.
               Set column to 0 for sum of column from i to i+every_other 
  """
  
  if arr.ndim == 1:
    arr = np.reshape(arr,(arr.size,1))
    #print (arr.shape)

  result = [] #Store result here
  extra_rows=[] #Store extra rows here
  if not col_options.any():
    col_options = np.zeros(arr.shape[1]) #Set colunn default
  for (i,line) in enumerate(arr): #Iterate rows
    if np.mod(i+1,every_other) == 0: #Check if it's row to sum over and store
      row = [] #Store row here
      for (j,ele) in enumerate(line):
        val = 0
        temp = []
        if col_options[j] == 0:
          for k in range(i-every_other+1,i+1):
            val+=arr[k,j]
          row.append(val)
      
        if col_options[j] == 1:
          for k in range(i-every_other+1,i+1):
            temp.append(arr[k,j])
          val = np.mean(temp)
          row.append(val)
        if col_options[j] == 2:
          for k in range(i-every_other+1,i+1):
            temp.append(arr[k,j])
          val = np.median(temp)

          row.append(val)
      result.append(row)
      extra_rows = []
    elif i == np.shape(arr)[0]-1:
     #print(i)
      extra_rows.append(line)
      return np.asarray(np.vstack((result,extra_rows))) #Returns extra rows
    else:
      extra_rows.append(line)
  return np.asarray(result)

#Equations


def distance(xyz1,xyz2):
  ds = np.zeros(3)
  for i in range(3):
    ds[i] = (xyz1[i]-xyz2[i])**2
  return sqrt(sum(ds))

def angles(xyz1,xyz2):
  ds = np.zeros(3)
  for i in range(3):
    ds[i] = abs(xyz2[i]-xyz1[i])
  theta_xy = arctan(ds[1]/ds[0])
  theta_xz = arctan(ds[2]/ds[0])
  return theta_xy,theta_xz  

def err_calc(true,pred):
  if isinstance(true,float) or isinstance(true,int) and isinstance(pred,float) or isinstance(pred,int):
    if true < 1e-10:
      return pred #Handle zeros
    else:
      return (pred-true)/true
  else:
    err = np.zeros(true.shape)
    for i in range(len(true)): #Should be same size
      if true[i] < 1e-10:
        err[i] = pred[i] #Handle zeros
      else:
        err[i] = (pred[i]-true[i])/true[i]
    return err

def print_stars():
  print('\n******************************\n')

def pickle_to_root(fpname):
  #Just type filename with or without path in front of it
  zz = 0

def get_central_arr(arr,nleft,nright=None):
  #Get center n values of array
  #nleft is number of values to left
  #nright is number of values to right
  if nright is None:
    nright = nleft
  length = len(arr)
  return arr[int(length/2)-nleft:int(length/2)+nright]

def isbetween(x,lower,upper): #Return true if x is between lower and upper
  if x > lower and x < upper:
    return True
  return False

def get_random_colors(n):
  color = []

  for i in range(n):
    color.append('#%06X' % randint(0, 0xFFFFFF))
  return color

