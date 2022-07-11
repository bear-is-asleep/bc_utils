import numpy as np
from random import choice
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from matplotlib.colors import Normalize
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
from numpy import sqrt,exp,arctan,cos,sin,pi,arccos
import uproot
from time import time

#Constants
hc = 1240 #eV nm
r_e = 2.818e-15 #m
alpha = 1/137 #alpha
m_e = .511e-3 #GeV/c^2
n_A = 1.3954/6.6335209e-23 #n_A/cm^3
m_u = 105.658 #MeV




#Get total POT
def get_pot(rootname,treename):
    #Input nueback dataframe
    POT_tree = uproot.open(f'{rootname}:{treename}/POT;1')
    pot = POT_tree.arrays('potbnb',library='np')
    return pot['potbnb'].sum()

#Return g4 dataframe
def get_g4_df(rootname,treename,offset=0):
    #Input root and tree name to return dataframe with g4 information
    #Offset run creates indexing for additional backgrounds
    g4_tree = uproot.open(f'{rootname}:{treename}/Event;1')
    with uproot.open(f'{rootname}:{treename}/Event;1') as file:
        keys = file.keys()
    g4_keys = keys[0:28]
    g4 = g4_tree.arrays(g4_keys,library='pd')
    runs = list(g4.loc[:,'run'])
    runs = [run+offset for run in runs] #Add offset to run value for additional background
    subruns = list(g4.loc[:,'subrun'])
    events = list(g4.loc[:,'event'])
    if offset != 0:
      arrays = [runs,subruns,events]
      tuples = list(zip(*arrays)) #Make multiindex
      index = pd.MultiIndex.from_tuples(tuples, names=['run','subrun','event'])
      g4 = g4.drop(['run','subrun','event'],axis=1)
      columns = g4.columns
      g4_data = g4.values
      g4 = pd.DataFrame(g4_data,index=index,columns=columns)
    else:
      g4 = g4.set_index(['run','subrun','event'])
    indeces = list(g4.index.values) 
    return g4,indeces

#Return genie dataframe
def get_genie_df(rootname,treename,branchname,offset=0):
    #Input root and tree name to return dataframe with genie information
    #Offset for different background types
    genie_tree = uproot.open(f'{rootname}:{treename}/{branchname};1')
    keys = genie_tree.keys()
    genie_keys = [key for key in keys if 'genie' in key]
    genie_keys.extend(['run','subrun','event','ccnc_truth'])
    genie = genie_tree.arrays(genie_keys,library='pd')
    if isinstance(genie,tuple):
      #assume only two dfs in tuple
      genie = pd.merge(left=genie[0],right=genie[1],how='outer')
    genie.loc[:,'run'] = genie.loc[:,'run']+offset
    genie = genie.set_index(['run','subrun','event'])
    indeces = list(genie.index.values) 
    genie = genie.sort_index()
    return genie,indeces

#Return dataframe keys
def get_keys(rootname,treename):
  with uproot.open(f'{rootname}:{treename}/Event;1') as file:
    keys = file.keys()
  return keys


def no_particle(g4,pdg=2212):
    #Input g4 dataframe, return indeces that don't contain the particle
    g4 = g4.sort_index()
    indeces = []
    index_g4 = np.unique(g4.index.values)
    for index in index_g4:
        if pdg not in list(g4.loc[index,'pdg']): #If there are no particles
            indeces.append(index)
    return indeces

def calc_T(df):
    #df is g4 df, see g4 keys
    #We're going to be including T and theta_e
    E = df['Eng']
    m = df['Mass']
    df['T'] = sqrt(E**2-m**2)

def number_events(df):
    #Return number of events in a dataframe
    return df.index.drop_duplicates().shape[0]

def get_pot_normalized_df(df,target_pot,pot,events,seed=420,pot_per_event=-1):
  #Return dataframe with randomly selected events, normalized to target pot
  #Also returns total number of events
  np.random.seed(seed)
  #print(int(np.round(target_pot/pot*events)),int(np.round(target_pot/pot_per_event)))
  if pot_per_event == -1:
    n_keep = int(np.round(target_pot/pot*events))
  else:
    n_keep = int(np.round(target_pot/pot_per_event))
  n_drop = events - n_keep
  #print(n_drop)
  if n_drop < 0:
    raise Exception('You need more events chief!')
  index = df.index.drop_duplicates()
  #print(n_drop)
  drop_indices = np.random.choice(index, n_drop, replace=False)
  return df.drop(drop_indices),n_keep

def get_eventype_count(df,pdg,pdg_key='genie_primaries_pdg'):
  #Returns number of events with a pdg
  #Also returns indeces with pdg
  #Also returns fraction of events with pdg
  indeces = df.index.drop_duplicates()
  indeces = list(indeces)
  events = len(indeces)
  cnt = 0 #Counter for events
  has_pdg = []
  for index in indeces:
    vals = df.loc[index,pdg_key].values
    #print(vals)
    if pdg in vals:
      cnt += 1
      has_pdg.append(index)
  return cnt,has_pdg,cnt/events

def calc_thetat(df,method=0,return_key='theta_t',theta_xz_key='theta_xz',theta_yz_key='theta_yz',
px_key='genie_Px',py_key='genie_Py',pz_key='genie_Pz'):
  #Calculate transverse angle (off z-axis) for given dataframe, appends it to dataframe
  #Method 0 is using angles
  #Method 1 is using momenta
  if return_key in df.columns:
    df.drop(return_key,axis=1)
    #print('Key already exists in dataframe silly!')
  if method == 0:
    #Get values of angles
    theta_xz = df.loc[:,theta_xz_key].values
    theta_yz = df.loc[:,theta_yz_key].values
    thetat = sqrt(theta_xz**2+theta_yz**2) #Not sure if this works, makes sense to me though
  elif method == 1:
    #Get momenta values
    px = df.loc[:,px_key].values
    py = df.loc[:,py_key].values
    pz = df.loc[:,pz_key].values
    thetat = arctan(sqrt(px**2+py**2)/pz)

  df.loc[:,return_key] = abs(thetat)
  return df
def calc_thetave(df,return_key='theta_ve',status_key='genie_status_code',E_key='genie_Eng',pdg_key='genie_primaries_pdg',
                method=0):
  #Use marina's (docDB:25206) method for calculating the angle, use the neutrino truth energy
  indeces = df.index.drop_duplicates()
  nu_pdgs = (12,-12,14,-14)
  for ind in indeces:
    row = df.loc[ind]
    #print(row)
    status = list(row[status_key])
    pdgs = list(row[pdg_key])
    Es = list(row[E_key])

    thetas = np.full(len(row),-9999.) #Label where neutrino and electron are

    #Dumby vars
    T_e = -9999
    E_e = -9999
    p_e = -9999
    E_nu = -9999
    #Check which index is the incoming neutirno
    for i,l in enumerate(np.array([status,pdgs,Es]).T):
      if l[0] == 0 and l[1] in nu_pdgs: #Incoming neutrino
        E_nu = l[2]
      elif l[0] == 1 and abs(l[1]) == 11: #Outgoing electron
        e_ind = i
        E_e = l[2]
        #print(row.iloc[i][E_key],m_e,sqrt(E_e**2-m_e**2))
        p_e = sqrt(E_e**2-m_e**2)
        T_e = E_e - m_e
    #print(E_nu,E_e,m_e,p_e,T_e,thetas)
    #print(arccos(T_e/p_e*(m_e/E_nu+1)))
    if T_e == -9999 or E_e == -9999 or p_e == -9999 or E_nu == -9999:
      #thetas[e_ind] = -9999
      print(f'Missing electron event for run {ind[0]} subrun {ind[1]} event {ind[2]}')
    else:
      if method == 0:
        te = arccos(T_e/p_e*(m_e/E_nu+1))
      elif method == 1: #These are identical
        te = arccos(E_e/p_e*(1-m_e*(1-T_e/E_nu)/E_e))
      thetas[e_ind] = te
      #print(arccos(T_e/p_e*(m_e/E_nu+1)))
    #print(thetas[4])
    df.loc[ind,return_key] = np.array(thetas)
  return df

def calc_Etheta(df,return_key='E_theta^2',E_key='genie_Eng',theta_t_key='theta_t'):
  #Calc E_etheta^2 for electrons
  if return_key in df.columns:
    df.drop(return_key,axis=1)
    #print('Key already exists in dataframe silly!')
  #Get values
  E = df.loc[:,E_key].values
  theta_t = df.loc[:,theta_t_key].values
  E_theta2 = E*theta_t**2
  df.loc[:,return_key] = E_theta2
  return df

def get_scat_type(df,pdg_key='genie_primaries_pdg',return_key='scat_type'):
  #Appends scat type to dataframe
  if return_key in df.columns:
    df.drop(return_key,axis=1)
    #print('Key already exists in dataframe silly!')
  indeces = df.index.drop_duplicates()
  indeces = list(indeces)
  types = [] #Scattering types
  for index in indeces:
    temp_df = df.loc[index]
    #Apply pdg check
    pdgs = temp_df.loc[:,pdg_key].values
    if 14 in pdgs: #nu mu
      types.extend(np.full(len(pdgs),0))
    elif 12 in pdgs: #nu e
      types.extend(np.full(len(pdgs),1))
    if -14 in pdgs: #nu bar mu
      types.extend(np.full(len(pdgs),2))
    elif -12 in pdgs: #nu bar e
      types.extend(np.full(len(pdgs),3))
  df.loc[:,return_key] = types
  return df

def get_signal_background(scat,back):
  #Get signal to background ratio
  #print(len(list(scat.index.drop_duplicates())),len(list(back.index.drop_duplicates())))
  #print(len(list(scat.index.drop_duplicates())),len(list(back.index.drop_duplicates())))
  return len(list(scat.index.drop_duplicates()))/len(list(back.index.drop_duplicates()))

def make_cuts(df,pdg_key='genie_primaries_pdg',method=0,Etheta=0.003,Etheta_key='E_theta^2'):
  #Make background cuts, returns cut index
  #Method 0: E theta^2 cut
  e_df = df[abs(df.loc[:,pdg_key]) == 11] #Filter by electron
  indeces = list(e_df.index.drop_duplicates()) #Get event indeces
  keep_indeces = [] #keep indeces if they pass the cut
  for index in indeces:
    if method == 0: #E theta^2 cut
      E_theta = e_df.loc[index,Etheta_key]
      #print(E_theta)
      #print(isinstance(E_theta,np.floating),type(E_theta),index)
      if isinstance(E_theta,np.floating):
        #print(index,E_theta,Etheta,E_theta < Etheta)
        if E_theta < Etheta: #Less than cutoff
          keep_indeces.append(index)
      else: #handle initial electron
        e_temp = e_df[e_df.loc[:,'genie_status_code']!=0] #Don't use initial electron
        if e_temp.loc[index,Etheta_key] < Etheta: #Less than cutoff
          #print(e_df.loc[index,Etheta_key].values[0] < Etheta,Etheta,e_df.loc[index,Etheta_key].values[0])
          keep_indeces.append(index)
  
  return df.loc[keep_indeces]

def get_electron_count(df,pdg_key='genie_primaries_pdg',return_key='e_count',drop_duplicates=False):
  #Get electron count for each event
  indeces = df.index.drop_duplicates()
  indeces = list(indeces)
  cnts=[] #Count number of electrons in each event
  drop_index = [] #Drop indeces with multiple electrons
  for index in indeces:
    temp_df = df.loc[index]
    #temp_df = temp_df[temp_df.loc[:,'genie_status_code']!=0] #Get rid of initial electron
    pdgs = list(abs(temp_df.loc[:,pdg_key].values))
    es = pdgs.count(11) #Number of electrons
    if drop_duplicates and es > 1:
      drop_index.append(index)
      continue
    else: 
      cnts.extend(np.full(len(pdgs),es)) #append number of electrons in each event
  if drop_duplicates:
    df = df.drop(drop_index)
  df.loc[:,return_key] = cnts
  return df

#Return shower dataframe
def get_shw_df(rootname,treename,offset=0):
    #Input root and tree name to return dataframe with g4 information
    #Offset run creates indexing for additional backgrounds
    shw_tree = uproot.open(f'{rootname}:{treename}/Event;1')
    with uproot.open(f'{rootname}:{treename}/Event;1') as file:
        keys = file.keys()
    shw_keys = [key for key in keys if 'shw' in key]
    print(shw_keys)
    shw = shw_tree.arrays(shw_keys,library='pd')
    runs = list(shw.loc[:,'run'])
    runs = [run+offset for run in runs] #Add offset to run value for additional background
    subruns = list(shw.loc[:,'subrun'])
    events = list(shw.loc[:,'event'])
    if offset != 0:
      arrays = [runs,subruns,events]
      tuples = list(zip(*arrays)) #Make multiindex
      index = pd.MultiIndex.from_tuples(tuples, names=['run','subrun','event'])
      shw = shw.drop(['run','subrun','event'],axis=1)
      columns = shw.columns
      shw_data = shw.values
      shw = pd.DataFrame(shw_data,index=index,columns=columns)
    else:
      shw = shw.set_index(['run','subrun','event'])
    indeces = list(shw.index.values) 
    return shw,indeces
  
def drop_initial_e(df,pdg_key='genie_primaries_pdg',p_key='genie_P',status_key='genie_status_code',small=1e-5,method=1):
  if method == 0:
    good_inds = ~((df[pdg_key] == 11) & (df[p_key] < small).values) #Has small momentum
  elif method == 1:
    good_inds = ~((df[pdg_key] == 11) & (df[status_key] == 0).values) #Is initial part.
  return df.loc[good_inds]

def cut_pdg_event_ak(ak,pdg,pdg_key='genie_primaries_pdg',eng_key='genie_Eng',E_threshold=0,exclude=True):
  #Exclude true to remove events with pdg, otherwise we'll include
  keep_inds = [] #Keep indeces, if exclude is false this becomes remove inds
  for ind,row in enumerate(ak):
    pdg_inds = [i for i,val in enumerate(row[pdg_key]) if abs(val) == pdg] #find locations of pdgs
    if len(pdg_inds) == 0: #This means there are none of this pdg, continue 
      keep_inds.append(ind)
      continue
    if len([x for x in row[eng_key][pdg_inds] if x >= E_threshold]) == 0: #An event exceeding threshold E
      keep_inds.append(ind)
  if ~exclude:
    remove_inds = keep_inds
    #keep_inds = [] #Clear just in case
    all_inds = list(range(len(ak)))
    keep_inds = [x for x in all_inds if x not in remove_inds] #Redefine keep_inds
  return ak[keep_inds]

def cut_ccnc_event_ak(ak,ccnc=0,ccnc_key='ccnc_truth'):
  #Keep only cc events if cc = 0, otherwise only keep nc events
  keep_inds = []
  for ind,row in enumerate(ak):
    if row[ccnc_key] == ccnc:
      keep_inds.append(ind)
  return ak[keep_inds]

def cut_pdg_event(df,pdg,pdg_key='genie_primaries_pdg',eng_key='genie_Eng',m_key='genie_mass',E_threshold=0,exclude=True,
                max_count=1,check_antiparticle=True):
  #Exclude true to remove events with pdg, otherwise we'll include
  #Max count is max number of events allowed
  drop_inds = []
  indeces = list(df.index.drop_duplicates()) #Get run info for each event
  prev_ind = indeces[0]
  cnt = 0
  start = time()
  for ind in indeces:
    row = df.loc[ind]
    #row.sort_index()
    row = row[row.loc[:,'genie_status_code'] == 1] #Check outgoing final state particles
    row = row[abs(row.loc[:,pdg_key]) == pdg] #Keep only correct pdg in event
    Es = row[eng_key].to_numpy()
    ms = row[m_key].to_numpy()
    Ts = Es-ms #Kinetic energy less than threshold
    #print(ind,Ts,Es,ms)
    #print(ind,[x for x in Ts if x >= E_threshold])
    if len([x for x in Ts if x >= E_threshold]) >= max_count: #An event exceeding threshold E
      drop_inds.append(ind)
    
    
  if not exclude: #Switch this to inds we want to keep
    df = df.loc[drop_inds]
  else:
    df = df.drop(drop_inds)
  return df

def find_hadron_activity(df,pdg_key='genie_primaries_pdg',drop=False,drop_type=0):
  #Determine if event has hadron activity, some events didn't
  #Drop type 0: drop events with no activity, type 1: drop events with activity
  indeces = list(df.index.drop_duplicates()) #Get run info for each event
  hadron_active = []
  drop_inds = []
  for i,ind in enumerate(indeces):
    row = df.loc[ind]
    pdgs = row[pdg_key].to_numpy()
    atoms = []
    for pdg in pdgs:
      #print(len(str(pdg)),str(pdg)[0])
      if len(str(pdg)) == 10 and str(pdg)[0] == str(1):
        atoms.append(pdg)
    if atoms[0] == atoms[1]:
      hadron_active.extend(np.full(len(row),0)) #No activity, probably nu e scattering
      if drop and drop_type == 0:
        drop_inds.append(ind)
    else:
      hadron_active.extend(np.full(len(row),1)) #Activity
      if drop and drop_type == 1:
        drop_inds.append(ind)
  df.loc[:,'hadron_activity'] = hadron_active
  return df.drop(drop_inds)

    
  


