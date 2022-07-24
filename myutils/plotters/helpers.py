import glob
from PIL import Image
import sys
bc_utils_path = '/Users/bearcarlson/python_utils/'
sbnd_utils_path = '/sbnd/app/users/brindenc/mypython'
sys.path.append(bc_utils_path)
from bc_utils.utils import pic,plotters
from bc_utils.myutils import pic as mypic
from bc_utils.myutils import plotters as myplotters
import os
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

#Constants
day = date.today().strftime("%Y_%m_%d")

#Functions
def make_image_gif(folder,root_name,duration=100):
  os.chdir(folder)
  #print(os.getcwd())
  fp_in = f'{root_name}*.jpg'
  fp_out = f'{root_name}.gif'
  #print(folder+fp_in)

  imgs = glob.glob(folder+fp_in)
  imgs = pic.sortstrings_numerically(imgs)
  frames = [Image.open(image) for image in imgs]
  img = frames[0]
  img.save(fp=fp_out,format='GIF',append_images=frames,save_all=True,duration=duration,loop=0)

def make_grid_gif(cwd,N=10,maxchoice=1,neighbors=1,gridsize=3,probs=None,
  cmap='gray',mode='circle',probtype=None,loop=True,duration=100):
  #Save folder
  folder = f'grids_{day}/{mode}/N{N}/gs{gridsize}/mc{maxchoice}_n{neighbors}_cm{cmap[:3]}/'
  choices = np.arange(0,maxchoice+1,step=1) #Color choices

  #Make it
  grid = mypic.make_grid(gridsize,mode=mode,choices=choices,probs=probs) #grid
  plt.figure()
  plt.imshow(grid,cmap=cmap)
  plt.axis('off');
  plt.title('0')
  plotters.save_plot('grid0',dpi=100,folder_name=folder)
  if loop:
    plotters.save_plot(f'grid{2*N}',dpi=100,folder_name=folder)
  plt.close()
  for i in range(N):
    grid = mypic.update_center_grid(grid,choices=choices,neighbors=neighbors,probtype=probtype)
    plt.figure()
    plt.imshow(grid,cmap=cmap)
    plt.axis('off');
    plt.title(f'{i+1}')
    plotters.save_plot(f'grid{i+1}',dpi=100,folder_name=folder)
    if loop:
      plotters.save_plot(f'grid{2*N-i+1}',dpi=100,folder_name=folder)
    plt.close()
  
  plot_folder = f'{cwd}/Plots/{folder}'
  #print(folder,plot_folder)
  #print(plot_folder)
  myplotters.make_image_gif(plot_folder,'grid',duration=duration)
  os.chdir(cwd)

