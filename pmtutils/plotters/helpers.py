import matplotlib.pyplot as plt
from random import choice
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
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
from bc_utils.utils import pic,plotters
#matplotlib.rcParams['axes.unicode_minus'] = False

xls = 16 #axis size
tls = 20 #title size
lls = 16 #legend size
tbs=14 #Text box size
#Small displacement for text
small = 5
matplotlib.rcParams['xtick.labelsize'] = 13
matplotlib.rcParams['ytick.labelsize'] = 13 #Set to same as axis size 
matplotlib.rcParams['axes.labelsize'] = 'large'
matplotlib.rcParams['legend.fontsize'] = 'large'
matplotlib.rcParams['axes.titlesize'] = 'x-large'
matplotlib.rcParams['figure.figsize'] = [9,7]


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

def save_plot(fname,fig=None,ftype='.jpg',dpi=500):
    day = date.today().strftime("%d_%m_%Y")
    if fig == None:
      plt.savefig(f'{fname}{ftype}',bbox_inches = "tight",dpi=dpi)
    else:
      fig.savefig(f'{fname}{ftype}',bbox_inches = "tight",dpi=dpi)
    #print(os.getcwd())
    os.system("mv " + fname + "* Plots/Plots_" +day+"/")

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

def make_lines(mark_borders=False):
  plt.axhline(y=62.5,linestyle='--',linewidth=3)
  plt.axhline(y=-62.5,linestyle='--',linewidth=3)
  plt.axvline(x=125,linestyle='--',linewidth=3)
  plt.axvline(x=250,linestyle='--',linewidth=3)
  plt.axvline(x=375,linestyle='--',linewidth=3)
  if mark_borders:
    plt.axvline(x=500,linestyle='--',linewidth=3)
    plt.axvline(x=0,linestyle='--',linewidth=3)
    plt.axhline(y=200,linestyle='--',linewidth=3)
    plt.axhline(y=-200,linestyle='--',linewidth=3)


def kde_1dhist(df,titles,xlabels,ylabels,keys,show,save,save_name,fit_gaus,trim_outliers):
  #Make kde plots
  for (i,title) in enumerate(titles):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot()
    arr = df[keys[i]].to_numpy()
    w = 0.2
    n = math.ceil((arr.max() - arr.min())/w) #set bin width
    counts,_ = np.histogram(arr,n)
    plt.hist(arr,fill=True,bins=n,label='Data')
    if trim_outliers[i]:
      arr = remove_outliers(arr,max_dev=0.06)
      w = 0.2
      n = math.ceil((arr.max() - arr.min())/w) #set bin width
      counts,_ = np.histogram(arr,n)
      plt.hist(arr,fill=True,bins=n,label='Fit Data')
    if fit_gaus[i]: #Fit curve to gaussian and plot results
      """x,popt,pcov = fit_gaussian(arr)
      perr = np.sqrt(np.diag(pcov))
      params = {'$A$':[f'{popt[0]:.2f}$\pm${perr[0]:.1e}',''],
                '$\mu$':[f'{popt[1]:.2f}$\pm${perr[1]:.1e}',''],
                '$\sigma$':[f'{popt[2]:.2f}$\pm${perr[2]:.1e}','']}
      """
      mu, std = norm.fit(arr)
      params = {'$\mu$':[f'{mu:.2f}',''],
                '$\sigma$':[f'{std:.2f}','']}
      s = convert_p_str(params)
      ax.text(0.75, 0.85,s,transform=ax.transAxes,bbox=dict(facecolor='Pink',alpha=0.5),horizontalalignment='left',fontsize=lls)
      # Plot the PDF.
      xmin, xmax = plt.xlim()
      x = np.linspace(xmin, xmax, int(1e5))
      p = norm.pdf(x, mu, std)
      scale = max(counts)/max(p)
      plt.plot(x, scale*p, 'k', linewidth=1,label='Gaussian Fit') 
      #ax.plot(x,gaussian(x,popt[0],popt[1],popt[2]),label='Gaussian fit')

    plt.xlabel(xlabels[i],fontsize=xls)
    plt.ylabel(ylabels[i],fontsize=xls)
    plt.title(titles[i],fontsize=tls)
    ax.legend(fontsize=lls)
    if i == 0 or i == 2:
      plt.xlim(-2, 5)
      ax.axvline(x=0,linestyle='-.',color='red')
    if i == 1:
      plt.xlim(-0.1, 4)
      ax.axvline(x=1,linestyle='-.',color='red')
    if save[i]:
      save_plot(save_name[i])
    if show[i]:
      plt.show()
    else:
      plt.close(fig)

def plot_TPC(tpc,label,label_title,df,coating=2,cmap='viridis',return_plot=False,
            normalize=False,facecolor='#98F5FF',det_key='ophit_opdet',mark_borders=False,
            make_colorbar=False,label_channels=True,invert_xaxis=False,small=small):
  #If coating is 2, plot both, coating 0 for coated, coating 1 for uncoated
  if df.shape[0] != 120:
    print('df needs to contain only one event, or combined events')
    return None
  if invert_xaxis:
    small = 1.2*small


  #plt.figure()
  #sns.distplot(df[label].to_numpy())
  #plt.show()
  #Plot 2d hist with colorscale as label
  fig = plt.figure(figsize=(10,8))
  ax = fig.add_subplot()
  ax.set_facecolor(facecolor)
  make_lines(mark_borders)



  data_points = []
  for _,line in df.iterrows():
    skip = False #Skips text for PMTs that are filtered
    det_type = int(line['ophit_opdet_type'])
    det_ch = str(line[det_key])
    x = line['ophit_opdet_x']
    y = line['ophit_opdet_y']
    z = line['ophit_opdet_z']
    c = line[label]
    data_line = [x,y,z,c]
    if tpc == 0 and x < 0:
      #Apply coating cut
      if coating == 2:
        if det_type == 0 or det_type == 1:
          data_points.append(data_line)
          skip = True #Keeps good PMTs
      elif coating == 1 or coating == 0:
        if det_type == coating:
          data_points.append(data_line)
          skip = True #Keeps good PMTs
      if label_channels:
        if z > 250 and skip:
          if det_ch == 166 or det_ch == 160:
            ax.text(z-2*small,y-2*small,det_ch,fontsize=tbs)
          else:
            ax.text(z-2*small,y+small,det_ch,fontsize=tbs)
        if z < 250 and skip:
          if det_ch == 150 or det_ch == 156:
            ax.text(z+0.15*small,y-2*small,det_ch,fontsize=tbs)
          else:
            ax.text(z+0.15*small,y+small,det_ch,fontsize=tbs)
    
    if tpc == 1 and x > 0: #186 included (for some reason)
      #Apply coating cut
      if coating == 2:
        if det_type == 0 or det_type == 1:
          data_points.append(data_line)
          skip = True #Keeps good PMTs
      elif coating == 1 or coating == 0:
        if det_type == coating:
          data_points.append(data_line)
          skip = True #Keeps good PMTs
      if label_channels:
        if z > 250 and skip: 
          if det_ch == 166 or det_ch == 160:
            ax.text(z-2*small,y-2*small,det_ch,fontsize=tbs)
          else:
            ax.text(z-2*small,y+small,det_ch,fontsize=tbs)
        if z < 250 and skip:
          if det_ch == 150 or det_ch == 156:
            ax.text(z+0.15*small,y-2*small,det_ch,fontsize=tbs)
          else:
            ax.text(z+0.15*small,y+small,det_ch,fontsize=tbs)

  data_points = np.asarray(data_points)
  #print(data_points[:,3])
  if normalize:
    if data_points[:,3].sum() != 0: #Don't divide if the sume is zero!
      data_points[:,3] = data_points[:,3]/data_points[:,3].sum()
  sc = ax.scatter(data_points[:,2],data_points[:,1],c=data_points[:,3],cmap=cmap,s=80,alpha=0.7)
  #ax.margins(x=0.05)
  if make_colorbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(sc,cax=cax,ax=ax)
  #cbar.set_label(f'{label_title}',rotation=270,fontsize=xls)
  ax.set_xlabel('Z [cm]',fontsize=xls)
  ax.set_ylabel('Y [cm]',fontsize=xls)
  #ax.set_title(f'{label_title} TPC{tpc}',fontsize = tls)
  if tpc == 0:
    ax.set_title(f'{label_title} West APA Back',fontsize = tls)
  elif tpc == 1:
    ax.set_title(f'{label_title} East APA Back',fontsize = tls)
  if invert_xaxis:
    ax.invert_xaxis()
  if return_plot:
    return fig,ax,sc
  else:
    return fig,ax

#Plot muon tracks
def plot_tracks(df,x1_key0,y1_key0,x2_key0,y2_key0,x1_key1,y1_key1,x2_key1,y2_key1,ax='None',fig='None',indeces=0,
  z1_key0=None,z2_key0=None,z1_key1=None,z2_key1=None,tpc=2):
  #Plot muon tracks with coordinate x as horizantel axis, y as vertical
  #Use keys

  #Handle text for 3rd coord
  plot_3d_coord = True
  if z1_key0 is None: #Handle plotting 3rd coordinate option
    plot_3d_coord = False

  #Set figure 
  if ax == 'None' and fig == 'None':
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    plot_text = True
  else: plot_text = False
  if isinstance(df[x1_key0],np.int64): #Why is it np.int64? Ask pandas idk
    #TPC0
    x1_0 = np.asarray([df[x1_key0]])
    y1_0 = np.asarray([df[y1_key0]])
    x2_0 = np.asarray([df[x2_key0]])
    y2_0 = np.asarray([df[y2_key0]])

    #TPC1
    x1_1 = np.asarray([df[x1_key1]])
    y1_1 = np.asarray([df[y1_key1]])
    x2_1 = np.asarray([df[x2_key1]])
    y2_1 = np.asarray([df[y2_key1]])

    if plot_3d_coord:
      z1_0 = np.asarray([df[z1_key0]])
      z2_0 = np.asarray([df[z2_key0]])
      z1_1 = np.asarray([df[z1_key1]])
      z2_1 = np.asarray([df[z2_key1]])
  else:
    x1_0 = df[x1_key0].to_numpy()
    y1_0 = df[y1_key0].to_numpy()
    x2_0 = df[x2_key0].to_numpy()
    y2_0 = df[y2_key0].to_numpy()
    #TPC1
    x1_1 = df[x1_key1].to_numpy()
    y1_1 = df[y1_key1].to_numpy()
    x2_1 = df[x2_key1].to_numpy()
    y2_1 = df[y2_key1].to_numpy()

    if plot_3d_coord:
      z1_0 = df[z1_key0].to_numpy()
      z2_0 = df[z2_key0].to_numpy()
      z1_1 = df[z1_key1].to_numpy()
      z2_1 = df[z2_key1].to_numpy()




  #Keep track of event ID

  #old method
  if indeces == 0:
    events = df['event'].to_numpy()
    subruns = df['subrun'].to_numpy()
    runs = df['run'].to_numpy()
  else:
    events = []
    subruns = []
    runs = []
    for index in indeces:
      events.append(index[2])
      subruns.append(index[1])
      runs.append(index[0])
    events = np.asarray(events)
    subruns = np.asarray(subruns)
    runs = np.asarray(runs)

  


  #Create labels for plot
  if 'x' in x1_key0:
    xlabel = 'x [cm]'
  if 'y' in x1_key0:
    xlabel = 'y [cm]'
  if 'z' in x1_key0:
    xlabel = 'z [cm]'
  if 'x' in y1_key0:
    ylabel = 'x [cm]'
  if 'y' in y1_key0:
    ylabel = 'y [cm]'
  if 'z' in y1_key0:
    ylabel = 'z [cm]'
  title = f'$\mu$ track trajectory'

  #print(events.shape,x1_1.shape)
  events0 = []
  subruns0 = []
  runs0 = []
  events1 = []
  subruns1 = []
  runs1 = []

  #Locally define small
  small = 10
  for i in range(events.shape[0]):
    #Small offset for text
    smallx = choice([-1*small,small,0])
    smally = choice([-1*small,small,0])
    #In TPC0
    if x1_1[i] == -999 and x2_1[i] == -999 and y1_1[i] == -999 and y2_1[i] == -999 and tpc == 0 or tpc == 2:
     #and events[i] not in events0:
      #Slope and intercept for text
      events0.append(events[i])
      subruns0.append(subruns[i])
      runs0.append(runs[i])
      xs = [x1_0[i],x2_0[i]]
      ys = [y1_0[i],y2_0[i]]
      xtext,ytext = (xs[1]+xs[0])/2+smallx,(ys[1]+ys[0])/2+smally
      if plot_text:
        ax.plot(xs,ys,ls='-')
        ax.set_xlabel(xlabel,fontsize=xls)
        ax.set_ylabel(ylabel,fontsize=xls)
        ax.set_title(title,fontsize=tls)
        #ax.text(xtext,ytext,f'{int(runs[i])},{int(subruns[i])},{int(events[i])}',fontsize=tbs)
        #ax.text(xtext,ytext,f'{int(subruns[i])},{int(events[i])}',fontsize=tbs)
      else: 
        ax.plot(xs,ys,ls='--',c='r',label=r'$\mu$ tracks')
        #ax.legend(fontsize=lls)
        #plt.text(xtext,ytext,f'{int(events[i])}',fontsize=tbs)
      if plot_3d_coord:
        zs = [z1_0[i],z2_0[i]]
        text1 = [xs[0]+smallx,ys[0]+smally]
        text2 = [xs[1]+smallx,ys[1]+smally]
        plt.text(text1[0],text1[1],f'x = {zs[0]:.1f}',fontsize=tbs)
        plt.text(text2[0],text2[1],f'x = {zs[1]:.1f}',fontsize=tbs)
    #In TPC1
    if x1_0[i] == -999 and x2_0[i] == -999 and y1_0[i] == -999 and y2_0[i] == -999 and tpc == 1 or tpc == 2:
      # and events[i] not in events1:
      #Slope and intercept for text
      events1.append(events[i])
      subruns1.append(subruns[i])
      runs1.append(runs[i])
      xs = [x1_1[i],x2_1[i]]
      ys = [y1_1[i],y2_1[i]]
      xtext,ytext = (xs[1]+xs[0])/2+smallx,(ys[1]+ys[0])/2+smally
      if plot_text:
        ax.plot(xs,ys)
        ax.set_xlabel(xlabel,fontsize=xls)
        ax.set_ylabel(ylabel,fontsize=xls)
        ax.set_title(title,fontsize=tls)
        #ax.text(xtext,ytext,f'{int(runs[i])},{int(subruns[i])},{int(events[i])}',fontsize=tbs)
        ax.text(xtext,ytext,f'{int(subruns[i])},{int(events[i])}',fontsize=tbs)
      else: 
        ax.plot(xs,ys,ls='--',c='r',label=r'$\mu$ tracks')
        #ax.legend(fontsize=lls)
        #plt.text(xtext,ytext,f'{int(events[i])}',fontsize=tbs)
      if plot_3d_coord:
        zs = [z1_1[i],z2_1[i]]
        text1 = [xs[0],ys[0]]
        text2 = [xs[1],ys[1]]
        print(text1,text2,zs)
        ax.text(text1[0],text1[1],f'x = {zs[0]:.1f}',fontsize=tbs)
        ax.text(text2[0],text2[1],f'x = {zs[1]:.1f}',fontsize=tbs)

  #save_plot('tracks')
  #plt.show()
  return ax,fig

def make_plane_meshes(x,yl=-200,yu=200,zl=0,zu=500,yiter=400,ziter=500,xiter=200,n_ref=1):
  ys = np.linspace(yl,yu,2) #Get all y refs.
  zs = np.linspace(zl,zu,2) #Get all z refs.
  yy,zz = np.meshgrid(ys,zs) #Mesh grid
  xx = np.full(yy.shape,x) #Populate mesh grid
  xs = np.linspace(-n_ref*xiter,+n_ref*xiter,2*n_ref+1) #Get all x refs.
  ys = np.linspace(-n_ref*yiter,yiter*n_ref,2*n_ref+1) #Get all y refs.
  zs = np.linspace(-ziter*n_ref,+ziter*n_ref,2*n_ref+1) #Get all z refs.
  #print(xx,yy,zz,xs,ys,zs)
  grids = []
  for xal in xs:
    for yal in ys:
      for zal in zs:
        grids.append([xx+xal,yy+yal,zz+zal,0]) #Add mesh grid to list
  return grids

def plot_ref(xkey,ykey,zkey,ckey,plane_grid,n_ref=1,x=None,y=None,z=None,
coord_ref_df=pd.DataFrame(),elev=20,azim=40,axis='on',title=''):
  if coord_ref_df.empty:
    if x != None and y != None and z != None:
      coord_ref_df = pic.single_plane_reflection(x,y,z,x_refs=n_ref,y_refs=n_ref,z_refs=n_ref,initialize=True)
    else: 
      raise Exception('We need some coordinates for this lame dataframe')
  if plane_grid:
    #print('Making Reflection inception')
    ttt = 0
  else:
    raise Exception('We need a plane grid')
      
  fig = plt.figure(figsize=(9,7))
  ax = Axes3D(fig,auto_add_to_figure=False)
  fig.add_axes(ax)

  x = coord_ref_df.loc[:,xkey]
  y = coord_ref_df.loc[:,ykey]
  z = coord_ref_df.loc[:,zkey]
  c = coord_ref_df.loc[:,ckey]

  if n_ref > 0:
    s = 200/n_ref
  else: 
    s= 200
  im = ax.scatter3D(x,z,y,c=c,s=s,cmap='viridis_r',edgecolors='black',alpha=0.6)
  for grid in plane_grid:
    plane_ref = ax.plot_surface(grid[0],grid[2],grid[1],alpha=0.3)
  #ax.axis('off')
  ax.view_init(elev,azim)
  if axis != 'off':
    fig.colorbar(im)
  else:
    ax.set(xlabel='x',ylabel='z',zlabel='y')
    ax.set_title(title,fontsize=tls+15)
  ax.axis(axis)
  return ax

def get_xy_bins(df,xkey,ykey,index,bw,tpc=2,pmt=2):
  #Make kde histogram using dataframe and its key
  #df is input dataframe
  #xkey is x axis key to make bin sizes
  #ykey is y axis to make scale
  #index is which event we want
  #bw is bin wdith
  #pmt specifies which pmt we're looking at: 0 is coated, 1 is uncoated, 2 is all of them, otherwise its the channel
  xs = df.loc[index,xkey].sort_index().values
  ys = df.loc[index,ykey].sort_index().values
  pmts = df.loc[index,'ophit_opch'].values
  coatings = df.loc[index,'ophit_opdet_type'].values
  tpcs = df.loc[index,'op_tpc'].values
  binedges = np.arange(xs.min(),xs.max()+bw,bw)

  check_all = False #Check all pmts will be using if pmt=2
  check_ch = False #Check only a single chanel
  if tpc == 2: 
    check_tpc = False
    title_tpc = ''
  else: 
    check_tpc = True
    title_tpc = f' TPC{tpc}'
  if pmt == 0 or pmt == 1: #Use this to 
    check_coating = True
    if pmt == 0:
      title_pmt = ' Coated PMTs '
    if pmt == 1:
      title_pmt = ' Uncoated PMTs '
  else:
    check_coating = False
    if pmt == 2:
      check_all = True
      title_pmt = ''
    else:
      check_ch = True
      title_pmt = f' PMT{pmt} '
  #Make title for plots
  #title = f'PE vs. timing{title_pmt}{title_tpc}\nRun {index[0]}, Subrun {index[1]}, Event {index[2]}'
  title = f'PE vs. timing{title_pmt}{title_tpc}'
  y_hist = np.zeros(len(binedges)+1) #Histogram values, add one buffer
  inds = np.digitize(xs,binedges) #Returns list with indeces where value belongs
  for i,ind in enumerate(inds):
    if check_tpc and tpc != tpcs[i]: continue #Skip events with incorrect tpc
    if check_coating and coatings[i] == pmt: #Make sure we're checking the right coating
      y_hist[ind] += ys[i] #Add y-val which belongs to this pmt
    elif check_all:
      y_hist[ind] += ys[i] #Add y-val which belongs to this pmt
    elif check_ch and pmts[i] == pmt: #Make sure we're checking the right pmt
      y_hist[ind] += ys[i] #Add y-val which belongs to this pmt
  #bincenters = binedges - bw/2 #temp fix of x-axis not being centered
  bincenters = binedges #temp fix of x-axis not being centered
  return bincenters,y_hist,title

def make_bar_scatter_plot(bincenters,yvals,bw,title='',left=-1,right=8,truncate=True,normalize=False,
  textx=0.45,texty=0.95,vlinex1=None,vlinex2=None):
  #Make figure and axis
  fig = plt.figure(figsize=(7,2.5))
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
  min_t = bincenters[0] - bw/2
  max_t = bincenters[-1] + bw/2
  max_bin = bincenters[np.where(yvals == max(yvals))][0] #Find t bin center with max PE
  if vlinex1 is None and vlinex2 is None:
    plot_lines = False
    parameters = {'Max PE ': f'{yvals.max():.2e}',
                r'Peak T ($\mu$s)':f'{max_bin:.3f}',
                #'Median PE = ': yvals.median(),
                #'Mean PE ': f'{yvals.mean():.2e}',
                'Total PE ': f'{yvals.sum():.2e}',
                r'$\Delta t$ ':f'{bw:.3f}'}
                #r'Time slice ($\mu$s) ': f'[{min_t:.1f},{max_t:.1f}]',
                #r'Time slice ($\mu$s) ': f'[{min_t:.1f},{max_t:.1f}]'}
  else:
    plot_lines = True
    parameters = {'Max PE ': f'{yvals.max():.2e}',
                r'Peak T ($\mu$s)':f'{max_bin:.3f}',
                #'Median PE = ': yvals.median(),
                #'Mean PE ': f'{yvals.mean():.2e}',
                'Total PE ': f'{yvals.sum():.2e}',
                r'$\Delta t$ ':f'{bw:.3f}',
                #r'Time slice ($\mu$s) ': f'[{min_t:.1f},{max_t:.1f}]',
                r'Time slice ($\mu$s) ': f'[{vlinex1},{vlinex2}]'}
  stattext = plotters.convert_p_str(parameters)
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  # place a text box in upper right in axes coords
  ax.text(textx, texty, stattext, transform=ax.transAxes, fontsize=tbs,
        verticalalignment='top', bbox=props)
  ax.set_title(title,fontsize=tls)
  ax.set_xlabel(r'$t$ ($\mu$s)',fontsize=xls)
  ax.set_ylabel('PE',fontsize=xls)
  if plot_lines:
    ax.axvline(x=vlinex1,ls='--',c='k')
    ax.axvline(x=vlinex2,ls='--',c='k')
  return ax,fig

def interactive_TPC(tpc,label,label_title,df,coating=2,cmap='viridis',return_plot=False,
            normalize=False,facecolor='cyan',ax=None,fig=None,sc=None,label_PMTs=True,vmax=None):
  #If coating is 2, plot both, coating 0 for coated, coating 1 for uncoated
  if df.shape[0] != 120:
    print('df needs to contain only one event, or combined events')
    return None
  #Plot 2d hist with colorscale as label
  if fig is None and ax is None:
    #plt.subplots_adjust(left=0.25, bottom=0.25)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot()
    make_lines()
  #if sc is not None:
  #  sc.remove()
  #fig = plt.figure(figsize=(10,8))
  #ax = fig.add_subplot()
  #make_lines()
  ax.set_facecolor(facecolor)
  



  data_points = []
  for _,line in df.iterrows():
    #print(line)
    skip = False #Skips text for PMTs that are filtered
    det_type = int(line['ophit_opdet_type'])
    det_ch = str(int(line['ophit_ch']))
    x = line['ophit_opdet_x']
    y = line['ophit_opdet_y']
    z = line['ophit_opdet_z']
    c = line[label]
    data_line = [x,y,z,c]
    if tpc == 0 and x < 0:
      #Apply coating cut
      if coating == 2:
        if det_type == 0 or det_type == 1:
          data_points.append(data_line)
          skip = True #Keeps good PMTs
      elif coating == 1 or coating == 0:
        if det_type == coating:
          data_points.append(data_line)
          skip = True #Keeps good PMTs
      if z > 250 and skip and label_PMTs:
        if det_ch == 166 or det_ch == 160:
          ax.text(z-2*small,y-2*small,det_ch,fontsize=tbs)
        else:
          ax.text(z-2*small,y+small,det_ch,fontsize=tbs)
      if z < 250 and skip and label_PMTs:
        if det_ch == 150 or det_ch == 156:
          ax.text(z+0.15*small,y-2*small,det_ch,fontsize=tbs)
        else:
          ax.text(z+0.15*small,y+small,det_ch,fontsize=tbs)
    
    if tpc == 1 and x > 0: #186 included (for some reason)
      #Apply coating cut
      if coating == 2:
        if det_type == 0 or det_type == 1:
          data_points.append(data_line)
          skip = True #Keeps good PMTs
      elif coating == 1 or coating == 0:
        if det_type == coating:
          data_points.append(data_line)
          skip = True #Keeps good PMTs
      if z > 250 and skip and label_PMTs: 
        if det_ch == 166 or det_ch == 160:
          ax.text(z-2*small,y-2*small,det_ch,fontsize=tbs)
        else:
          ax.text(z-2*small,y+small,det_ch,fontsize=tbs)
      if z < 250 and skip and label_PMTs:
        if det_ch == 150 or det_ch == 156:
          ax.text(z+0.15*small,y-2*small,det_ch,fontsize=tbs)
        else:
          ax.text(z+0.15*small,y+small,det_ch,fontsize=tbs)

  data_points = np.asarray(data_points)
  #print(data_points[:,3])
  if normalize:
    if data_points[:,3].sum() != 0: #Don't divide if the sume is zero!
      data_points[:,3] = data_points[:,3]/data_points[:,3].sum()
  #print(data_points)
  sc = ax.scatter(data_points[:,2],data_points[:,1],c=data_points[:,3],cmap=cmap,s=80,alpha=0.7,vmax=vmax)
  plt.subplots_adjust(left=0.25)
  ax.margins(x=0.05)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  cbar = fig.colorbar(sc,cax=cax,ax=ax)
  #cbar.set_label(f'{label_title}',rotation=270,fontsize=xls)
  ax.set_xlabel('Z [cm]',fontsize=xls)
  ax.set_ylabel('Y [cm]',fontsize=xls)
  ax.set_title(f'{label_title} TPC{tpc}',fontsize = tls)
  ax.set_xlim([0,500])
  ax.set_ylim([-200,200])
  if return_plot:
    return fig,ax,sc,cax
  else:
    return fig,ax

def plot_g4_muon(g4,muon,index,thigh,tlow,small=20,x='z',y='y',save_fig=False,display_run_info=True,n=50):
  #g4 and muon are the dataframes for this info
  #index is (run,subrun,event)
  #thigh and tlow are treadout maximum and maximum (see get_treadout in pmtutils)
  #x and y are axes to plot
  #Run info
  run = index[0]
  subrun = index[1]
  event = index[2]
  
  if display_run_info:
    run_info = f'Run: {run} Subrun: {subrun} Event: {event}'
  else:
    run_info = ''

  fig = plt.figure(figsize=(9,7))
  ax = fig.add_subplot()

  #Keep single event
  gtemp = g4.loc[index,:]
  mtemp = muon.loc[index,:]

  #Get list of random colors from utils
  color = pic.get_random_colors(n) #Default to 50

  #Populate g4
  gxs = [[]]
  gys = [[]]
  if isinstance(gtemp,pd.core.series.Series):
    use_iloc = False #Determines if we need to use ilov, only use on dataframe (not series)
    gxs.append([gtemp[f'StartPoint{x}'],gtemp[f'EndPoint{x}']])
    gys.append([gtemp[f'StartPoint{y}'],gtemp[f'EndPoint{y}']])
  else:
    for _,row in gtemp.iterrows():
      gxs.append([row[f'StartPoint{x}'],row[f'EndPoint{x}']])
      gys.append([row[f'StartPoint{y}'],row[f'EndPoint{y}']])
  gxs.pop(0); gys.pop(0) #Delete extra

  #Populate muons
  mxs = [[]]
  mys = [[]]
  if isinstance(mtemp,pd.core.series.Series): #theres only one track
    mxs.append([mtemp[f'muontrk_{x}1'],mtemp[f'muontrk_{x}2']])
    mys.append([mtemp[f'muontrk_{y}1'],mtemp[f'muontrk_{y}2']])
  else:
    for line,row in mtemp.iterrows(): #iterate through all tracks
      mxs.append([row[f'muontrk_{x}1'],row[f'muontrk_{x}2']])
      mys.append([row[f'muontrk_{y}1'],row[f'muontrk_{y}2']])
  mxs.pop(0); mys.pop(0)
  for i in range(len(mxs)):#iterate through all tracks, labeling one
    if i ==0:
      ax.plot(mxs[i],mys[i],linewidth=10,alpha=0.5,label='Muon Tracks')
    else:
      ax.plot(mxs,mys,linewidth=10,alpha=0.5)
    ax.text(mxs[i][0],mys[i][0],f'{mtemp["muontrk_type"]:.0f}',fontsize=14) #track type label
  for i in range(len(gxs)): #iterate through all g4 tracks, lableing one
    if i  == 0:
      ax.plot(gxs[i],gys[i],ls='--',c=color[i],label='G4 Tracks')
    else:
      ax.plot(gxs[i],gys[i],ls='--',c=color[i])
    if use_iloc:
        ax.scatter(gtemp.iloc[i][f'det{x}'],gtemp.iloc[i][f'det{y}'],s=100,marker='s',c=color[i])
    else:
      ax.scatter(gtemp[f'det{x}'],gtemp[f'det{y}'],s=100,marker='s',c=color[i])
    

  ax.legend(fontsize=14)
  ax.set_xlabel(f'{x} [cm]',fontsize=16)
  ax.set_ylabel(f'{y} [cm]',fontsize=16)

  ax.set_title(r'Tracks $t_r \in$' + f'[{tlow},{thigh}] ms\n'+run_info,fontsize=20)
  if x == 'x':
    ax.axvline(0,linewidth=11,color='black')
    ax.text(-50,100,r'CPA$\rightarrow$',fontsize=16)
    ax.set_xlim([-200-small,200+small]);
  if y == 'x':
    ax.axhline(0,linewidth=11,color='black')
    ax.text(100,-50,r'CPA$\uparrow$',fontsize=16)
    ax.set_ylim([-200-small,200+small]);
  if x == 'z':
    ax.set_xlim([0-small,500+small])
  if y == 'y':
    ax.set_ylim([-200-small,200+small]);
  if y == 'z':
    ax.set_ylim([0-small,500+small]);
  if save_fig:
    plotters.save_plot(f'g4_muontrks_{y}{x}_event{run:.0f}{subrun:.0f}{event:.0f}')
    plt.close()





