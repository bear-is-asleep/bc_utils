import matplotlib.pyplot as plt
import numpy as np
from bc_utils.utils import pic,plotters

xls = 16 #axis size
tls = 20 #title size
lls = 16 #legend size
tbs=14 #Text box size

def make_bar_scatter_plot(bincenters,yvals,bw,title='',left=0,right=2000,truncate=True,normalize=False):
  #Make figure and axis
  fig = plt.figure(figsize=(10,3))
  ax = fig.add_subplot()
  #Truncate data to capture interesting stuff
  if truncate:
    boolean_array = np.logical_and(bincenters >= left, bincenters <= right) #find indeces between left and right
    inds_in_range = np.where(boolean_array)[0]
    bincenters=bincenters[inds_in_range]
    yvals=yvals[inds_in_range]
  if normalize:
    yvals = yvals/yvals.sum()

  #ax.plot(bincenters,yvals,c='r')
  ax.bar(bincenters,yvals,width=bw)

  #Make parameters dictionary
  min_w = bincenters[0] - bw
  max_w = bincenters[-1] + bw
  parameters = {'Max Q ': f'{yvals.max():.2e}',
                #'Median PE = ': yvals.median(),
                #'Mean Q ': f'{yvals.mean():.2e}',
                'Total Q ': f'{yvals.sum():.2e}',
                'Wire binwidth':bw}
                #r'Wire slice': f'[{min_w:.1f},{max_w:.1f}]'}
  stattext = plotters.convert_p_str(parameters)
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  # place a text box in upper right in axes coords
  ax.text(0.05, 0.95, stattext, transform=ax.transAxes, fontsize=tbs,
        verticalalignment='top', bbox=props)
  ax.set_title(title,fontsize=tls)
  ax.set_xlabel('Wire Number',fontsize=xls)
  ax.set_ylabel('Q (ADC)',fontsize=xls)
  ax.grid()
  return ax,fig
