import numpy as np
from bc_utils.utils import pic


#Make a defined grid
def make_grid(n,mode='random',choices=[0,1],probs=None):
  #mode is condition of starting
  #center: single point in senter
  #random: randomize using probs (NONE is uniform)
  #right side: start descending from right to left
  #zigzag: diagonal like line
  #rollacross: rotate colors every row
  #rollup: rotate colors every column

  length = 2*n+1
  grid = np.zeros((length,length))

  if mode == 'random':
    for i in range(length):
      for j in range(length):

        grid[i,j] = np.random.choice(a=choices,p=probs) #Random choice
  elif mode == 'rightside':
    cols = np.linspace(choices[0],choices[-1],length)
    cols = [round(x) for x in cols] #Round to nearest int for colormap
    for i in range(length):
      for j in range(length):
        grid[i,j] = cols[j] #right side desending
  elif mode == 'center':
    colors = np.linspace(choices[0],choices[-1],length)
    colors = [round(x) for x in colors] #Round to nearest int for colormap
    for c in range(length):
      for r in range(length):
        #distance from center in x and y
        dx = abs(c-n)
        dy = abs(r-n)
        #print(c,r,length,dx,dy,2*n)
        grid[c][r]=colors[dx+dy]
  elif mode == 'zigzag':
    colors = np.linspace(choices[0],choices[-1],length)
    colors = [round(x) for x in colors] #Round to nearest int for colormap
    for c in range(length):
      for r in range(length):
        #distance from center in x and y
        dx = c-n
        dy = r-n
        grid[c][r]=colors[dx+dy]
  elif mode == 'circle':
    colors = np.linspace(choices[0],choices[-1],length)
    colors = [round(x) for x in colors] #Round to nearest int for colormap
    for c in range(length):
      for r in range(length):
        #distance from center in x and y
        dx = abs(c-n)
        dy = abs(r-n)
        #radial distance
        dr = np.sqrt(dx**2+dy**2) #cnt goes down from the center
        #print(c,r,length,dx,dy,2*n)
        grid[c][r]=colors[round(dr)]
  elif mode == 'rollacross':
    colors = np.linspace(choices[0],choices[-1],length)
    colors = [round(x) for x in colors] #Round to nearest int for colormap
    for r in range(length):
      grid[:][r] = np.roll(colors,r)
  elif mode == 'rollup':
    colors = np.linspace(choices[0],choices[-1],length)
    colors = [round(x) for x in colors] #Round to nearest int for colormap
    for c in range(length):
      grid[c][:] = np.roll(colors,c)
  else:
    grid[n][n] = choices[-1]
  return grid

#Update grid
def update_center_grid(grid,choices=[0,1],neighbors=1,probtype=None):#Randomly update points in grid
  grid_copy = grid.copy() #Copy grid
  if probtype == 'gaussian':
    arr = pic.discrete_gauss(len(choices))
  for i in range(len(grid_copy)):
    for j in range(len(grid_copy)):
      cnts=np.zeros(len(choices)+1) #cnt how many nearest neighbors are 0,1,exist
      neighbor_checks = np.arange(-1*neighbors,neighbors+1,step=1) #check these inds, including 0
      for q in neighbor_checks:
        for qq in neighbor_checks:#randomly update neares blocks
          x_ind = q+i #x index
          y_ind = qq+j #y index
          if pic.isbetween(x_ind,-1,len(grid_copy)) and pic.isbetween(y_ind,-1,len(grid_copy)): #check edges
            if q!=0 or qq!=0:
              cnts[-1]+=1 #Update total count
              for ind,choice in enumerate(choices):
                #print(ind,choice,grid_copy[q+i,qq+j],x_ind,y_ind)
                if grid_copy[x_ind,y_ind] == choice:
                  cnts[ind]+=1
      for ind,choice in enumerate(choices):
        if probtype=='gaussian':
          probs = np.zeros(len(cnts)-1) #Length of probs array
          neighbor_arr = cnts[:-1]/float(cnts[-1]) #Get probs of neighbors being certain choice
          for nind,prob in enumerate(neighbor_arr):
            gaus = np.roll(arr,nind) #roll gaussian over by choices amount
            probs+= prob*gaus #average gaussian out
        else:
          probs = cnts[:-1]/float(cnts[-1])
        if grid_copy[i][j] == choice:
          grid[i][j] = np.random.choice(a=choices,p=probs)

  return grid

