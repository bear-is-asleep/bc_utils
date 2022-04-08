import numpy as np
import pandas as pd

def get_xy_bins(df,xkey,ykey,index,bw,tpc=2,plane=3):
  #df is input dataframe
  #xkey is x axis key to make bin sizes
  #ykey is y axis to make scale
  #index is which event we want
  #bw is bin wdith
  #plane specifies collection plane 0:U 1:V 2:Y (z-axis)
  xs = df.loc[index,xkey].values
  ys = df.loc[index,ykey].values
  planes = df.loc[index,'hit_plane'].values
  tpcs = df.loc[index,'hit_tpc'].values
  binedges = np.arange(xs.min(),xs.max()+bw,bw)

  if tpc == 2: 
    check_tpc = False
    title_tpc = ''
  else:
    check_tpc = True
    title_tpc = f' TPC{tpc}'
  if plane <= 2:
    check_plane = True
    title_plane = f' Plane {plane} '
  else:
    check_plane = False
    title_plane = ' All Planes '
  #Make title for plots
  title = f'Q vs. Wire Number{title_plane}{title_tpc}\nRun {index[0]}, Subrun {index[1]}, Event {index[2]}'

  y_hist = np.zeros(len(binedges)) #Histogram values
  inds = np.digitize(xs,binedges) #Returns list with indeces where value belongs
  for i,ind in enumerate(inds):
    if check_tpc and tpc != tpcs[i]: continue #Skip events with incorrect tpc
    if check_plane and plane != planes[i]: continue #skip these too
    y_hist[ind] += ys[i] #Add y-val which belongs to this x-val

  #bincenters = binedges - bw/2 #temp fix of x-axis not being centered
  bincenters = binedges #temp fix of x-axis not being centered
  return bincenters,y_hist,title

