import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import os
from datetime import date
import sys
bc_utils_path = '/Users/bearcarlson/python_utils/'
sbnd_utils_path = '/sbnd/app/users/brindenc/mypython'
sys.path.append(bc_utils_path) #My utils path
from bc_utils.pmtutils import plotters as pmtplotters
from scipy import optimize
from sklearn.linear_model import LinearRegression
from numpy import isin, sqrt,exp,arctan,cos,sin,pi
from time import time
from bc_utils.utils import pic

#Constants
hc = 1240 #eV nm
r_e = 2.818e-15 #m
alpha = 1/137 #alpha
m_e = 511e3 #eV/c^2
n_A = 1.3954/6.6335209e-23 #n_A/cm^3
m_u = 105.658 #MeV
efficiency = 0.25 #Efficiency of light collection
efficiency_TPB = 0.5 #Efficiency of TPB conversion
l_att = 66 #cm https://arxiv.org/pdf/1611.02481.pdf 

#Constants

#Centers of tpc
tpc0_xyz = np.array([-100,0,250])
tpc1_xyz = np.array([100,0,250])
r = r_pmt = 19/2 #cm https://www.hamamatsu.com/us/en/product/type/R5912/index.html
V = 4*5*2 #TPC volume m^3
V_cm = V*1e6
A = 4*5*2 + 2*4*2 + 2*5*2 #TPC area m^2
A_cm = A*1e4
L = 6*V_cm/A_cm #Characteristic length of detector cm

l_R = 55#Rayleigh scattering length cm +-5 https://pure.royalholloway.ac.uk/portal/files/29369028/2018GraceEPHD.pdf
l = 63 #Attenuation length cm +- 3 https://www.mpi-hd.mpg.de/gerda/public/2020/phd2020_BirgitZatschler.pdf_compressed.pdf
QE = 0.25 # cathode at 390 nm https://www.hamamatsu.com/us/en/product/type/R5912/index.html
Tw = 0.92 #Transmissivity of PMT https://www.azom.com/article.aspx?ArticleID=4765
Rw = 0 #Reflectivity of PMT

#Rs = 0.886 #Reflectivity of surface coating

E_ws = 0.5 #Assume same as PMT coating
e_ws = 0.5 #Shifting efficiency TPB
Ref_cov = 4*5/A #Area of inner surface covered by TPB coated reflective surfaces
dEdx = 2.2 #MeV/cm check later

#MAPPING to installation labels
channel_mapping = {
  6: ['C','C33'], #Bottom right
  60: ['C','C31'], #Bottom left
  36: ['C','C22'], #Center
  62: ['C','C11'], #Top left
  8: ['C','C13'], #Top right
  10: ['B','B33'], #Bottom right
  64: ['B','B31'], #Bottom left
  38: ['B','B22'], #Center
  66: ['B','B11'], #Top left
  12: ['B','B13'], #Top right
  14: ['A','A33'], #Bottom right
  68: ['A','A31'], #Bottom left
  40: ['A','A22'], #Center
  70: ['A','A11'], #Top left
  16: ['A','A13'], #Top right
  84: ['F','F33'], #Bottom right
  138: ['F','F31'], #Bottom left
  114: ['F','F22'], #Center
  140: ['F','F11'], #Top left
  86: ['F','F13'], #Top right
  88: ['E','E33'], #Bottom right
  142: ['E','E31'], #Bottom left
  116: ['E','E22'], #Center
  144: ['E','E11'], #Top left
  90: ['E','E13'], #Top right
  92: ['D','D33'], #Bottom right
  146: ['D','D31'], #Bottom left
  118: ['D','D22'], #Center
  148: ['D','D11'], #Top left
  94: ['D','D13'], #Top right
  162: ['I','I33'], #Bottom right
  216: ['I','I31'], #Bottom left
  192: ['I','I22'], #Center
  218: ['I','I11'], #Top left
  164: ['I','I13'], #Top right
  166: ['H','H33'], #Bottom right
  220: ['H','H31'], #Bottom left
  194: ['H','H22'], #Center
  222: ['H','H11'], #Top left
  168: ['H','H13'], #Top right
  170: ['G','G33'], #Bottom right
  224: ['G','G31'], #Bottom left
  196: ['G','G22'], #Center
  226: ['G','G11'], #Top left
  172: ['G','G13'], #Top right
  240: ['L','L33'], #Bottom right
  294: ['L','L31'], #Bottom left
  270: ['L','L22'], #Center
  296: ['L','L11'], #Top left
  242: ['L','L13'], #Top right
  244: ['K','K33'], #Bottom right
  298: ['K','K31'], #Bottom left
  272: ['K','K22'], #Center
  300: ['K','K11'], #Top left
  246: ['K','K13'], #Top right
  248: ['J','J33'], #Bottom right
  302: ['J','J31'], #Bottom left
  274: ['J','J22'], #Center
  304: ['J','J11'], #Top left
  250: ['J','J13'], #Top right
  #X - O
  #W N  V M  S P  T Q  U R
  61: ['X','X33'], #Bottom right
  7: ['X','X31'], #Bottom left
  37: ['X','X22'], #Center
  9: ['X','X11'], #Top left
  63: ['X','X13'], #Top right
  65: ['W','W33'], #Bottom right
  11: ['W','W31'], #Bottom left
  39: ['W','W22'], #Center
  13: ['W','W11'], #Top left
  67: ['W','W13'], #Top right
  69: ['V','V33'], #Bottom right
  15: ['V','V31'], #Bottom left
  41: ['V','V22'], #Center
  17: ['V','V11'], #Top left
  71: ['V','V13'], #Top right
  139: ['U','U33'], #Bottom right
  85: ['U','U31'], #Bottom left
  115: ['U','U22'], #Center
  87: ['U','U11'], #Top left
  141: ['U','U13'], #Top right
  143: ['T','T33'], #Bottom right
  89: ['T','T31'], #Bottom left
  117: ['T','T22'], #Center
  91: ['T','T11'], #Top left
  145: ['T','T13'], #Top right
  147: ['S','S33'], #Bottom right
  93: ['S','S31'], #Bottom left
  119: ['S','S22'], #Center
  95: ['S','S11'], #Top left
  149: ['S','S13'], #Top right
  217: ['R','R33'], #Bottom right
  163: ['R','R31'], #Bottom left
  193: ['R','R22'], #Center
  165: ['R','R11'], #Top left
  219: ['R','R13'], #Top right
  221: ['Q','Q33'], #Bottom right
  167: ['Q','Q31'], #Bottom left
  195: ['Q','Q22'], #Center
  169: ['Q','Q11'], #Top left
  223: ['Q','Q13'], #Top right
  225: ['P','P33'], #Bottom right
  171: ['P','P31'], #Bottom left
  197: ['P','P22'], #Center
  173: ['P','P11'], #Top left
  227: ['P','P13'], #Top right
  295: ['O','O33'], #Bottom right
  241: ['O','O31'], #Bottom left
  271: ['O','O22'], #Center
  243: ['O','O11'], #Top left
  297: ['O','O13'], #Top right
  299: ['N','N33'], #Bottom right
  245: ['N','N31'], #Bottom left
  273: ['N','N22'], #Center
  247: ['N','N11'], #Top left
  301: ['N','N13'], #Top right
  303: ['M','M33'], #Bottom right
  249: ['M','M31'], #Bottom left
  275: ['M','M22'], #Center
  251: ['M','M11'], #Top left
  305: ['M','M13'] #Top right
  #X - O
  #W N  V M  S P  T Q  U R
}

# channel_mapping = {6: ['C', 'C33'],
#  60: ['C', 'C31'],
#  36: ['C', 'C22'],
#  62: ['C', 'C11'],
#  8: ['C', 'C13'],
#  10: ['B', 'B33'],
#  64: ['B', 'B31'],
#  38: ['B', 'B22'],
#  66: ['B', 'B11'],
#  12: ['B', 'B13'],
#  14: ['A', 'A33'],
#  68: ['A', 'A31'],
#  40: ['A', 'A22'],
#  70: ['A', 'A11'],
#  16: ['A', 'A13'],
#  84: ['F', 'F33'],
#  138: ['F', 'F31'],
#  114: ['F', 'F22'],
#  140: ['F', 'F11'],
#  86: ['F', 'F13'],
#  88: ['E', 'E33'],
#  142: ['E', 'E31'],
#  116: ['E', 'E22'],
#  144: ['E', 'E11'],
#  90: ['E', 'E13'],
#  92: ['D', 'D33'],
#  146: ['D', 'D31'],
#  118: ['D', 'D22'],
#  148: ['D', 'D11'],
#  94: ['D', 'D13'],
#  162: ['I', 'I33'],
#  216: ['I', 'I31'],
#  192: ['I', 'I22'],
#  218: ['I', 'I11'],
#  164: ['I', 'I13'],
#  166: ['H', 'H33'],
#  220: ['H', 'H31'],
#  194: ['H', 'H22'],
#  222: ['H', 'H11'],
#  168: ['H', 'H13'],
#  170: ['G', 'G33'],
#  224: ['G', 'G31'],
#  196: ['G', 'G22'],
#  226: ['G', 'G11'],
#  172: ['G', 'G13'],
#  240: ['L', 'L33'],
#  294: ['L', 'L31'],
#  270: ['L', 'L22'],
#  296: ['L', 'L11'],
#  242: ['L', 'L13'],
#  244: ['K', 'K33'],
#  298: ['K', 'K31'],
#  272: ['K', 'K22'],
#  300: ['K', 'K11'],
#  246: ['K', 'K13'],
#  248: ['J', 'J33'],
#  302: ['J', 'J31'],
#  274: ['J', 'J22'],
#  304: ['J', 'J11'],
#  250: ['J', 'J13'],
#  7: ['X', 'X33'],
#  61: ['X', 'X31'],
#  37: ['X', 'X22'],
#  63: ['X', 'X11'],
#  9: ['X', 'X13'],
#  11: ['W', 'W33'],
#  65: ['W', 'W31'],
#  39: ['W', 'W22'],
#  67: ['W', 'W11'],
#  13: ['W', 'W13'],
#  15: ['V', 'V31'],
#  69: ['V', 'V33'],
#  41: ['V', 'V22'],
#  71: ['V', 'V13'],
#  17: ['V', 'V11'],
#  85: ['U', 'U31'],
#  139: ['U', 'U33'],
#  115: ['U', 'U22'],
#  141: ['U', 'U13'],
#  87: ['U', 'U11'],
#  89: ['T', 'T31'],
#  143: ['T', 'T33'],
#  117: ['T', 'T22'],
#  145: ['T', 'T13'],
#  91: ['T', 'T11'],
#  93: ['S', 'S31'],
#  147: ['S', 'S33'],
#  119: ['S', 'S22'],
#  149: ['S', 'S13'],
#  95: ['S', 'S11'],
#  163: ['R', 'R31'],
#  217: ['R', 'R33'],
#  193: ['R', 'R22'],
#  219: ['R', 'R13'],
#  165: ['R', 'R11'],
#  167: ['Q', 'Q31'],
#  221: ['Q', 'Q33'],
#  195: ['Q', 'Q22'],
#  223: ['Q', 'Q13'],
#  169: ['Q', 'Q11'],
#  171: ['P', 'P31'],
#  225: ['P', 'P33'],
#  197: ['P', 'P22'],
#  227: ['P', 'P13'],
#  173: ['P', 'P11'],
#  241: ['O', 'O31'],
#  295: ['O', 'O33'],
#  271: ['O', 'O22'],
#  297: ['O', 'O13'],
#  243: ['O', 'O11'],
#  245: ['N', 'N31'],
#  299: ['N', 'N33'],
#  273: ['N', 'N22'],
#  301: ['N', 'N13'],
#  247: ['N', 'N11'],
#  249: ['M', 'M31'],
#  303: ['M', 'M33'],
#  275: ['M', 'M22'],
#  305: ['M', 'M13'],
#  251: ['M', 'M11']}


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


def truncate_df(df,predkeys=[''],errkeys=[''],keys_to_sum = {''},
keys_to_median = {''},mod_chs=False):
  #If mod_chs, we take the ch%1000, so that reflected channels convert to the same channel 
  #Truncate df for average values across all events and runs, len(df) = number of PMTs
  cols = df.shape[1]
  col_options = np.ones(cols)
  if mod_chs:
    df.loc[:,'ophit_ch'] = df.loc[:,'ophit_ch'].values%1000 #Convert all channels into one

  #set sum_combine to 0 for sum, 1 for mean, 2 for med
  for i,key in enumerate(df.keys()):
    if key in keys_to_sum:
      col_options[i]=0
    if key in keys_to_median:
      col_options[i]=2
  #Extract all PMT channel information and store in single df
  all_trunc = []
  for i in range(int(df['ophit_ch'].min()),int(df['ophit_ch'].max())+1):
    df_temp = df.loc[df['ophit_ch']==i]
    if df_temp.empty:
      continue
    np_temp = df_temp.to_numpy()
    np_trunc = sum_combine(np_temp,every_other=np_temp.shape[0],col_options=col_options)
    all_trunc.append(np_trunc)
  all_trunc = np.squeeze(all_trunc)
  df = pd.DataFrame(data=all_trunc,columns=df.keys())
  if len(predkeys)>0:
    obs = df.loc[:,'ophit_obs']
    predkeys.sort()
    errkeys.sort()
    for i in range(len(predkeys)): #Calc prediction directly from dataframe predictions (fixes error)
      errkeys[i]
      preds = df.loc[:,predkeys[i]].values
      df.loc[:,errkeys[i]] = err_calc(obs,preds)
  return  df

#Equations
def N_0 (length,dEdx):
  #Number of ionized photons given in SBND proposal paper
  ionization_rate = 2.4e4 #gamma/MeV
  return length*dEdx*ionization_rate 
def d_calc(x):
  #Dispersion of photons do to radiating spherically
  if np.isscalar(x): #Calculating dispersion for scalar
    if x < sqrt(r_pmt**2/2):
      #zz = 0
      return 1/2
    elif x >= sqrt(r_pmt**2/2):
      return r_pmt**2/(4*x**2)
      #zz = 0
    #return 1 #We're going to try this for now
  else: #Calculating dispersion for vector x input, ds are dispersion values
    ds = []
    for val in x:
      if val < sqrt(r_pmt**2/2):
        ds.append(1/2)
      elif val >= sqrt(r_pmt**2/2):
        ds.append(r_pmt**2/(4*val**2))
    return np.asarray(ds)
    #return np.ones(x.shape) #Try this for now

def d_calc_lazy(x,f=pi*r_pmt**2/A_cm):
  #Dispersion of photons, radiating in a box
  #Account for infinite reflections by multiplying by constant Rs
  #f is fractional coverage of PMT
  #R is reflectivity, this will be the area of the reflective surface divided by the total area
  #coating is type of PMT coating

  if np.isscalar(x): #Calculating dispersion for scalar
    return f #We're going to try this for now
  else: #Calculating dispersion for vector x input, ds are dispersion values
    return np.full(x.shape,f) #Try this for now

def I_calc(x,l):
  #Photons lost due to scattering
  if not np.isscalar(x):
    vals = []
    for val in x:
      vals.append(exp(-val/l))
    return np.asarray(vals)
  else:
    return exp(-x/l)
def N_calc(x,efficiency,length,dEdx,l,photon_interactions=True,dispersion=True):
  #Photons that reach PMT as a function of x (distance from PMT)
  if photon_interactions and dispersion:
    return N_0(length,dEdx)*d_calc(x)*I_calc(x,l)*efficiency
  elif photon_interactions and not dispersion:
    return N_0(length,dEdx)*I_calc(x,l)*efficiency
  elif not photon_interactions and dispersion:
    return N_0(length,dEdx)*d_calc(x)*efficiency
  else: 
    return N_0(length,dEdx)*efficiency
def N_calc_lazy(x,efficiency,length,dEdx,l):
  #Photons that reach PMT as a function of x (distance from PMT)
  #Use lazy d method
  return N_0(length,dEdx)*d_calc_lazy(x)*I_calc(x,l)*efficiency

def total_hits_PMT(df):
  #Returns dataframe which combines arrays to find total ophits
  arr = df.values #Get values into numpy array
  col_options = []
  for key in df.keys():
    if key == 'nophits':
      col_options.append(0)
    else:
      col_options.append(2)

  new_arr = sum_combine(arr,every_other=arr.shape[0],col_options=col_options)

  return pd.DataFrame(new_arr,columns=df.keys())

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

def A_calc(a,b):
  return pi*a*b
def a_calc(d,r,theta_xz):
  return d*r*cos(theta_xz)/(d+r*sin(theta_xz))
def b_calc(d,r,theta_xy):
  return d*r*cos(theta_xy)/(d+r*sin(theta_xy))

def get_pmts(opdet):
  #Take opdet in and return pmt information
  opdet = opdet.drop(['nophits','ophit_peakT','ophit_width','ophit_area','ophit_amplitude','ophit_pe'],axis=1)
  opdet = opdet.drop_duplicates()
  pmts_coated = opdet[opdet['ophit_opdet_type'] == 0] #0:coated 1:uncoated
  pmts_uncoated = opdet[opdet['ophit_opdet_type'] == 1] #0:coated 1:uncoated
  pmts = pd.concat((pmts_coated,pmts_uncoated))

  temp_df = pd.DataFrame(index=pmts.index,columns=['opdet_tpc','opdet_area','f','distance'])

  for row,line in pmts.iterrows():
    x = line['ophit_opdet_x']
    y = line['ophit_opdet_y']
    z = line['ophit_opdet_z']
    xyz = np.array([x,y,z])
    if x < 0:
      tpc = 0
      xyz2 = tpc0_xyz
    elif x > 0:
      tpc = 1
      xyz2 = tpc1_xyz
    temp_df.loc[row]['opdet_tpc'] = tpc
    d = distance(xyz,xyz2) #Distance from center to PMTs
    phi_xy,phi_xz = angles(xyz,xyz2) #Angle from PMT
    theta_xy = abs(phi_xy)
    theta_xz = abs(phi_xz) 
    a = a_calc(d,r,theta_xz) #Compression in z-axis, right left
    b = b_calc(d,r,theta_xy) #Compression in y-axis, up down
    A = abs(A_calc(a,b)) #Area of resulting elipse
    temp_df.loc[row]['opdet_area'] = A
    temp_df.loc[row]['f'] = A/A_cm
    temp_df.loc[row]['distance'] = d
  print(pmts.shape)
  pmts = pd.concat((pmts,temp_df),axis=1)
  #pmts = pmts.loc[:,~pmts.columns.duplicated()] #Drop duplicate columns
  return pmts

def err_calc(true,pred):
  if isinstance(true,float) or isinstance(true,int) and isinstance(pred,float):
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


def distances_to_PMT(hits,x1,y1,z1,x2,y2,z2,xp,yp,zp):
	#Distance of all track points to PMT given track
  hits = int(hits) #Convert to hits
  d = sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2) #Total distance of track
  lengths = np.array([x2-x1,y2-y1,z2-z1]) #length of line segment each dimension
  step_sizes = lengths/hits
  d_to_PMT = [] #distance to PMT for each hit
  #Hit locations along line
  x_h = []
  y_h = []
  z_h = []
  for i in range(hits+1):
    x_h.append(x1+i*step_sizes[0])
    y_h.append(y1+i*step_sizes[1])
    z_h.append(z1+i*step_sizes[2])
    d_to_PMT.append(sqrt((x_h[i]-xp)**2+(y_h[i]-yp)**2+(z_h[i]-zp)**2)) #3D distance to PMT
  
  d_to_PMT = np.asarray(d_to_PMT)
  return d_to_PMT,x_h,y_h,z_h,np.linalg.norm(step_sizes) #Option to return hit loctations 

def print_stars():
  print('\n******************************\n')

def get_muon_tracks(pmt_df):
  #Return dataframe for plotting muon tracks in both TPCs
  columns=['muontrk_x1_0', 'muontrk_y1_0', 'muontrk_z1_0', 'muontrk_x2_0',
       'muontrk_y2_0', 'muontrk_z2_0','muontrk_x1_1', 'muontrk_y1_1', 'muontrk_z1_1', 'muontrk_x2_1',
       'muontrk_y2_1', 'muontrk_z2_1','event','subrun','run']
  mtrks_data = np.full((pmt_df.shape[0],len(columns)),-999)
  cnter = 0 #Use this to index
  for row,line in pmt_df.iterrows():
    i = cnter #event
    mtrks_data[i,12] = row[2] #Mark event
    mtrks_data[i,13] = row[1] #Mark subrun
    mtrks_data[i,14] = row[0] #Mark run

    if line['muontrk_tpc'] == 0: #Append for x1_0 - z2_0
        mtrks_data[i,0] = line['muontrk_x1']
        mtrks_data[i,1] = line['muontrk_y1']
        mtrks_data[i,2] = line['muontrk_z1']
        mtrks_data[i,3] = line['muontrk_x2']
        mtrks_data[i,4] = line['muontrk_y2']
        mtrks_data[i,5] = line['muontrk_z2']
    if line['muontrk_tpc'] == 1: #Append for x1_1 - z2_1
        mtrks_data[i,6] = line['muontrk_x1']
        mtrks_data[i,7] = line['muontrk_y1']
        mtrks_data[i,8] = line['muontrk_z1']
        mtrks_data[i,9] = line['muontrk_x2']
        mtrks_data[i,10] = line['muontrk_y2']
        mtrks_data[i,11] = line['muontrk_z2']
    cnter+=1


  mtrks = pd.DataFrame(mtrks_data,columns=columns).drop_duplicates()
  #Don't set indeces for this dataframe, it'll break the plotter code. Maybe I'll fix it tho
  return mtrks

def reflect_xz(y,flip=2,ref=1,L=400):
  #Returns new y, reflected across face in xz plane in detector
  #flip divisible by 2 is off bottom, otherwise off top
  #ref is number of reflections
  #L Length of y-coordinate
  y_prime=0

  if flip % 2 == 0:
    y = y+200 #Adjust such that xz plane starts at y=0
    y_prime = -y
    for n in range(2,ref+1):
      y_prime = -(y_prime+(n-1)*L)-(n-1)*L
  else: 
    y = y+200 #Adjust such that xz plane starts at y=0
    y_prime=y
    for n in range(1,ref+1):
      y_prime = -(y_prime-n*L)+n*L
  return y_prime-200

def reflect_yz(x,flip=2,ref=1):
  #Returns new y, reflected across face in xz plane in detector
  #flip divisible by 2 is off cpa, otherwise off apa (where PDS is)
  #ref is number of reflections
  L = 200 #Length of x-coordinate
  x_prime=0

  if flip % 2 == 0: #cpa
    cpa_ref = int((ref+1)/2)
    x_prime = x-L*ref #Since x is always the same
  else: #apa
    cpa_ref = int(ref/2)
    x_prime = x+L*ref #Since x is always the same
  return x_prime,cpa_ref,ref-cpa_ref #Also return number of cpa reflections, it has reflective coating

def reflect_xy(z,flip=2,ref=1,L=500):
  #Returns new z, reflected across face in xy plane in detector
  #flip divisible by 2 is off z=0, otherwise off z=500
  #ref is number of reflections
  #L Length of z-coordinate
  z_prime=0

  if flip % 2 == 0: #z=0
    z_prime = -z
    for n in range(2,ref+1):
      z_prime = -(z_prime+(n-1)*L)-(n-1)*L
  else: 
    z_prime=z
    for n in range(1,ref+1):
      z_prime = -(z_prime-n*L)+n*L
  return z_prime

def single_plane_reflection(x,y,z,flip_coord='xyz',x_refs=1,y_refs=1,z_refs=1,initialize=False,ch=0):
  #For single reflections only, this can happen 6 different ways:
  #Front-back,left-right,top-bottom
  #Default to returning all 6 types of reflections for set of coordinates, return cpa/apa ref. for each
  #Also default to 1 reflection along each dimension
  if initialize:
    if ch == 0:
      coord_ref = [[x,y,z,0,0,0]] #6 indexes for 3 coords. total # reflections, cpa ref., apa ref.
    else:
      coord_ref = [[x,y,z,0,0,0,ch]] #7 indexes for 3 coords. total # reflections, cpa ref., apa ref., ch #
  else: 
    coord_ref = []
  if 'xyz' in flip_coord: #Build other cases later if needed
    for x_ref in range(1,x_refs+1):
      for y_ref in range(1,y_refs+1):
        for z_ref in range(1,z_refs+1):
          x1,cpa_ref1,apa_ref1 = reflect_yz(x,ref=x_ref)
          x2,cpa_ref2,apa_ref2 = reflect_yz(x,flip=3,ref=x_ref)
          z1 = reflect_xy(z,ref=z_ref)
          z2 = reflect_xy(z,flip=3,ref=z_ref)
          y1 = reflect_xz(y,ref=y_ref)
          y2 = reflect_xz(y,flip=3,ref=y_ref)
          #Now there's 8+10+6=26 cases x,y,z,xy,xz,yz,xyz
          xs = np.array([x,x1,x2])
          ys = np.array([y,y1,y2])
          zs = np.array([z,z1,z2])
          for i,xal in enumerate(xs): 
            for j,yal in enumerate(ys):
              for k,zal in enumerate(zs): #Find all possible combinations of 3 xs, 3 ys, 3 zs
                if i == 0 and j == 0 and k == 0:
                  continue #skip events that don't reflect
                tot_ref = 0 #Get total number of reflections
                cpa_ref = 0 #cpa ref
                apa_ref = 0 #apa ref
                if i != 0:
                  tot_ref += x_ref
                  cpa_ref = 0
                  apa_ref = 0
                if j != 0:
                  tot_ref += y_ref
                if k != 0:
                  tot_ref += z_ref
                if i == 1:
                  cpa_ref = cpa_ref1
                  apa_ref = apa_ref1
                if i == 2:
                  cpa_ref = cpa_ref2
                  apa_ref = apa_ref2
                #6 indexes for 3 coords. total # reflections, cpa ref., apa ref.
                if ch == 0:
                  coord_ref.append([xal,yal,zal,tot_ref,cpa_ref,apa_ref])
                else: 
                  ch+=1000
                  coord_ref.append([xal,yal,zal,tot_ref,cpa_ref,apa_ref])
  if ch == 0:
    df = pd.DataFrame(coord_ref,columns=['x','y','z','tot_ref','cpa_ref','apa_ref']).drop_duplicates()
  else: 
    df = pd.DataFrame(coord_ref,columns=['ophit_opdet_x','ophit_opdet_y','ophit_opdet_z',
    'tot_ref','cpa_ref','apa_ref','ophit_opdet']).drop_duplicates()
  return df

def lin_fit_3d(pmt_hits_df,xkey='mhits',ykey='mean_distance',zkey='ophit_obs'):
  #Fit x,y to linear fit in 3 dimenstions
  pmt_hits_df_coated0 = pmt_hits_df[pmt_hits_df.loc[:,'ophit_opdet_type'] == 0]
  pmt_hits_df_coated1 = pmt_hits_df[pmt_hits_df.loc[:,'ophit_opdet_type'] == 1]
  dfs = [pmt_hits_df_coated0,pmt_hits_df_coated1]
  coefs = [] #xkey is first, then ykey
  bs = [] #z intercept of 3d plot 
  scores = [] #Model score on itself
  models = []
  for df in dfs:
    xx = df.loc[:,[xkey,ykey]].values
    y = df.loc[:,zkey].values
    model = LinearRegression()
    model.fit(xx,y)
    coefs.append(model.coef_)
    bs.append(model.intercept_)
    scores.append(model.score(xx,y))
    models.append(model)
  return coefs,bs,scores,models #Returns all of these u know..

def check_functionality(df,std=4):
  #Accept values up to std outside of 0 (no error)
  obs = df.loc[:,'ophit_obs']
  pred = df.loc[:,'ophit_pred']
  err = df.loc[:,'pred_err']
  calc_err = err_calc(obs,pred)
  std = np.std(err)
  temp = np.zeros(df.shape[0])
  for i,val in enumerate(calc_err):
    if abs(val) > 4*std:
      temp[i] = 0
    else: 
      temp[i] = 1

  df.loc[:,'functioning'] = temp
  return df

def fake_bad_pmt(df,chs,reduce=0.0):
  #Multiply given channels by reduction factor
  temp_df = df.copy() #Make a copy, not to lose info of original df
  bad_indeces = []
  for ch in chs:
    bad_indeces.append(temp_df[temp_df['ophit_ch'] == ch].index.values[0])
  temp_df.loc[bad_indeces,'ophit_obs'] *= reduce
  pred = temp_df.loc[:,'ophit_pred'].values
  obs = temp_df.loc[:,'ophit_obs'].values
  new_err = err_calc(obs,pred)
  temp_df.loc[:,'pred_err'] = new_err
  return temp_df

def get_peakT(op_df,pmts,tpc,index,bw,number_pmts=10,thresholdPE=1,convert_to_edge=False):
  #Must be an op_df
  #pmts are the pmt information
  #tpc specifies which one we want to check, 1 or 0
  #Get peak time, so number_pmts has to be above the threshold
  pmts = pmts[pmts.loc[:,'opdet_tpc']==tpc] #Iterate over correct channels
  chs = pmts.loc[:,'ophit_opdet'].values
  chPEs = [] #List of all channels PEs
  for ch in chs: #Get PE/timing for each channel
    bincenters,chPE,_ = pmtplotters.get_xy_bins(op_df,'ophit_peakT','ophit_pe',index,bw,pmt=ch,tpc=tpc) 
    chPEs.append(chPE)
  #Now iterate over the rows, to find when number_pmts exceeds the threshold
  chPEs = np.array(chPEs) #Convert to numpy array to have access to transpose
  for i,PEslice in enumerate(chPEs.T):#iterae of transpose to get each timeslice, instead of each channel
    if len([*filter(lambda x: x >= thresholdPE, PEslice)])>=number_pmts:
      peakT = bincenters[i]
      if convert_to_edge:
        return peakT - bw/2 #Convert to edge, not center
      else: 
        return peakT
  #return -9999 #Return if the peakT threshold is not found

def find_cosmicentrance(df,return_keyx='detx',return_keyy='dety',return_keyz='detz',x_key='StartPointx',y_key='StartPointy',z_key='StartPointz',
  theta_yx_key='theta_yx',theta_xz_key='theta_xz',theta_yz_key='theta_yz',E_key='Eng',p_key='P',maxx=200,maxy=200,maxz=500,minx=-200,miny=-200,minz=0,
  endx_key='EndPointx',endy_key='EndPointy',endz_key='EndPointz',method=0): #Find where g4 particle enters detector (cosmic)
  #Get vars
  xs = df.loc[:,x_key] #cm
  ys = df.loc[:,y_key]
  zs = df.loc[:,z_key]
  endxs = df.loc[:,endx_key] #cm
  endys = df.loc[:,endy_key]
  endzs = df.loc[:,endz_key]
  if method == 0: #Use momentum to get angles
    theta_yxs = df.loc[:,theta_yx_key]
    theta_yzs = df.loc[:,theta_yz_key]
    theta_xzs = df.loc[:,theta_xz_key]
  elif method == 1:
    theta_xzs = np.arctan((endxs-xs)/(endzs-zs))
    theta_yzs = np.arctan((endys-ys)/(endzs-zs))
    theta_yxs = np.arctan((endys-ys)/(endxs-xs))

  #Initialize variables
  t_drifts = np.zeros(xs.shape)
  t_enters = np.zeros(xs.shape)
  endxs = np.zeros(xs.shape)
  endys = np.zeros(xs.shape)
  endzs = np.zeros(xs.shape)
  found_cosmic = False

  for i in range(len(xs)):
    endy = 200
    theta_yx = theta_yxs.iloc[i]
    theta_yz = theta_yzs.iloc[i]
    theta_xz = theta_xzs.iloc[i]
    x = xs.iloc[i]
    y = ys.iloc[i]
    z = zs.iloc[i]

    endz = z + (endy-y)*(1/np.tan(theta_yz))
    endx = x + (endy-y)*(1/np.tan(theta_yx))
    #print(endx,endy,endz,pic.isbetween(endz,minz,maxz) and pic.isbetween(endx,minx,maxx))
    if pic.isbetween(endz,minz,maxz) and pic.isbetween(endx,minx,maxx): #Cosmic entered detector at top
      found_cosmic = True
      endxs[i] = endx
      endys[i] = endy
      endzs[i] = endz
    else: #Cosmic entered in sides
      #print('here')
      #Temporary endpoint checks on 4 side faces
      tempxs = np.full(4,-9999)
      tempys = np.full(4,-9999)
      tempzs = np.full(4,-9999)
      #Check which face it hits
      tempxs[0] = minx
      tempxs[1] = maxx
      tempzs[2] = minz
      tempzs[3] = maxz
      for j in range(4): #Iterate over all cases, select one with largest y-value
        if j < 2:
          tempzs[j] = z + (tempxs[j]-x)*(1/np.tan(theta_xz))
          tempys[j] = y + (tempxs[j]-x)*np.tan(theta_yx)
          #print(y + (tempxs[j]-x)*np.tan(theta_yx),j,i,x,y,z,tempxs[j],tempys[j],tempzs[j],theta_xz,theta_yx,theta_yz)
          if pic.isbetween(tempzs[j],minz,maxz) and pic.isbetween(tempys[j],miny,maxy): #Found an interaction
            found_cosmic = True
          else: #Set back to dumby values
            tempxs[j] = -9999
            tempys[j] = -9999
            tempzs[j] = -9999
            ttt = 0
        else:
          tempxs[j] = x + (tempzs[j]-z)*np.tan(theta_xz)
          tempys[j] = y + (tempzs[j]-z)*np.tan(theta_yz)
          #print(y + (tempzs[j]-z)*np.tan(theta_yz),j)
          if pic.isbetween(tempxs[j],minx,maxx) and pic.isbetween(tempys[j],miny,maxy): #Found an interaction
            found_cosmic = True
          else: #Set back to dumby values
            tempxs[j] = -9999
            tempys[j] = -9999
            tempzs[j] = -9999
            ttt = 0
      index = np.argmax(tempys) #Index with max y value, assume this is entrance point
      #Save entrance points with largest y-value
      endxs[i] = tempxs[index]
      endys[i] = tempys[index]
      endzs[i] = tempzs[index]
  #Save entrance points and return dataframe
  #print(found_cosmic)
  df.loc[:,return_keyx] = endxs
  df.loc[:,return_keyy] = endys
  df.loc[:,return_keyz] = endzs
  return df

def get_treadout(df,return_key='treadout',x_key='detx',t_key='StartT'): #Need to use find_cosmicentrance to user this
  #Get readout time of g4 track
  v_drift = 0.00016 #cm/ns
  t = df.loc[:,t_key]
  x = df.loc[:,x_key]
  df.loc[:,return_key] = t + x/v_drift
  return df

def get_xy_bins(df,xkey,ykey,index,bw,tpc=2,pmt=-1,xmin=None,xmax=None,arapucas=True):
  #Make kde histogram using dataframe and its key
  #df is input dataframe
  #xkey is x axis key to make bin sizes
  #ykey is y axis to make scale
  #index is which event we want
  #bw is bin wdith
  #pmt specifies which pmt we're looking at: 
  #0 is coated, (remove support for this with inclusion of x-arapucas)
  #1 is uncoated, (remove support for this with inclusion of x-arapucas)
  #-1 is all of them, 
  #otherwise its the channel (need to include x-arapucas soon)
  xs = df.loc[index,xkey].sort_index().values
  ys = df.loc[index,ykey].sort_index().values
  pmts = df.loc[index,'ophit_opch'].values
  coatings = df.loc[index,'ophit_opdet_type'].values
  tpcs = df.loc[index,'op_tpc'].values
  if xmax == None and xmin == None:
    binedges = np.arange(xs.min(),xs.max()+bw,bw)
    trim_edges = False
  else:
    binedges = np.arange(xmin,xmax+2*bw,bw)
    trim_edges = True #Digitize keeps excess data in extra bins, first and last ones. We should drop these since they're out of the region of interest

  check_all = False #Check all pmts will be using if pmt=2
  check_ch = True #Check only a single chanel
  if tpc == 2: 
    check_tpc = False
    title_tpc = ''
  else: 
    check_tpc = True
    title_tpc = f' TPC{tpc}'
  if not arapucas: #not checking arapucas
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
  else:
    check_ch = True #This will be the main option in pad, I'll fix this later if needed
    check_coating = False
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
  #print(y_hist,bincenters)
  if trim_edges:
    #Delete first and last elements
    bincenters = np.delete(bincenters,0)
    bincenters = np.delete(bincenters,-1)
    y_hist = np.delete(y_hist,0)
    y_hist = np.delete(y_hist,-1)
  return bincenters,y_hist,title



