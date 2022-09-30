import numpy as np
from random import choice
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import date
import sys
from time import time
import uproot

#Constants
hc = 1240 #eV nm
r_e = 2.818e-15 #m
alpha = 1/137 #alpha
m_e = .511e-3 #GeV/c^2
n_A = 1.3954/6.6335209e-23 #n_A/cm^3
m_u = 105.658 #MeV

#Key labels
hdrkeys = ['rec.hdr.run','rec.hdr.subrun','rec.hdr.evt'] #Header keys
primprefix = 'rec.mc.nu.prim.' #Add this to beginning of string for flat cafana for primary particls
mcnuprefix = 'rec.mc.nu.' #Add this to beginning of string for flat cafana for neutrino events
recoprefix = 'rec.reco.' #Add this to beginning of string for flat cafana for neutrino events
shwprefix = recoprefix+'shw.'
trkprefix = recoprefix+'trk.'

def get_pot_normalized_df(df,target_pot,pot,events,seed=420,pot_per_event=-1):
  """
  Return dataframe with randomly selected events, normalized to target pot
  Also returns total number of events

  :param param1: this is a first param
  :param param2: this is a second param
  :returns: this is a description of what is returned
  :raises keyError: raises an exception
  """
  
  np.random.seed(seed)
  n_keep = int(np.round(target_pot/pot*events))
  n_drop = events - n_keep
  if n_drop < 0:
    raise Exception('You need more events chief!')
  index = df.index.drop_duplicates()
  drop_indices = np.random.choice(index, n_drop, replace=False)
  return df.drop(drop_indices),n_keep

def get_pot_normalized_indeces(indeces,target_pot,pot,seed=420):
  """Returns list of indeces for target pot
  Returns dataframe if it's provided"""
  np.random.seed(seed)
  events = len(indeces)
  n_keep = int(np.round(target_pot/pot*events))
  n_drop = events - n_keep
  if n_drop < 0:
    raise Exception('You need more events chief!')
  drop_indeces = np.random.choice(indeces, n_drop, replace=False)
  return drop_indeces

def get_df_dropindeces(df,drop_indeces):
  """Since not all dfs have all indeces, we will use this to drop events from pot normalization
  Only drops events that are already in dataframe. Without this method we'd reach an out of bounds error"""
  indeces = df.index.drop_duplicates() #Get event list
  drop = [] #Make array to drop indeces
  for ind in drop_indeces:
    if ind in indeces:
      drop.append(ind)
  return df.drop(drop)

def get_df_keepindeces(df,keep_indeces):
  """Since not all dfs have all indeces, we will use this to keep events from pot normalization
  Only keeps events that are already in dataframe. Without this method we'd reach an out of bounds error"""
  indeces = list(df.index.drop_duplicates()) #Get event list
  keep_indeces = list(keep_indeces)
  common_indeces = list(set(indeces).intersection(keep_indeces)) #keep indeces in common
  return df.loc[common_indeces]



def calc_thetat(df,return_key=f'{primprefix}thetal',px_key=f'{primprefix}genp.x',
  py_key=f'{primprefix}genp.y',pz_key=f'{primprefix}genp.z'):
  """
  Calculates transverse angle from momentum
  """
  if return_key in df.columns:
    df.drop(return_key,axis=1)
    print('Key already exists in dataframe silly! - Ill replace it tho :)')
  #Get momenta values
  px = df.loc[:,px_key].values
  py = df.loc[:,py_key].values
  pz = df.loc[:,pz_key].values
  thetat = np.arctan(np.sqrt(px**2+py**2)/pz)

  df.loc[:,return_key] = abs(thetat)
  return df

def calc_Etheta(df,return_key=f'{primprefix}Etheta2',E_key=f'{primprefix}genE',
  theta_l_key=f'{primprefix}thetal'):
  """
  Calculates Etheta2 for all particles, useable for nu e scattering cut
  """
  if return_key in df.columns:
    df.drop(return_key,axis=1)
    print('Key already exists in dataframe silly! - Ill replace it tho :)')
  #Get values
  E = df.loc[:,E_key].values
  theta_t = df.loc[:,theta_l_key].values
  E_theta2 = E*theta_t**2
  df.loc[:,return_key] = E_theta2
  return df

def get_signal_background(scat,back):
  """Get signal to background ratio"""
  return len(list(scat.index.drop_duplicates()))/len(list(back.index.drop_duplicates()))

def make_cuts(df,pdg_key=f'{primprefix}pdg',status_key=f'{primprefix}gstatus',method=0,
  Etheta=0.003,Etheta_key=f'{primprefix}Etheta2'):
  """make etheta2 cuts"""
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
        e_temp = e_df[e_df.loc[:,status_key]!=0] #Don't use initial electron
        if e_temp.loc[index,Etheta_key] < Etheta: #Less than cutoff
          #print(e_df.loc[index,Etheta_key].values[0] < Etheta,Etheta,e_df.loc[index,Etheta_key].values[0])
          keep_indeces.append(index)
  
  return df.loc[keep_indeces],keep_indeces

def make_reco_cuts(df,pdg_key=f'{shwprefix}razzle.pdg',method=0,
  Etheta=0.003,Etheta_key=f'{shwprefix}Etheta2'):
  """make etheta2 cuts"""
  #Make background cuts, returns cut index
  #Method 0: E theta^2 cut
  e_df = df[abs(df.loc[:,pdg_key]) == 11] #Filter by electron
  indeces = list(e_df.index.drop_duplicates()) #Get event indeces
  keep_indeces = [] #keep indeces if they pass the cut
  for index in indeces:
    if method == 0: #E theta^2 cut
      E_theta = e_df.loc[index,Etheta_key]
      if isinstance(E_theta,np.floating):
        if E_theta < Etheta: #Less than cutoff
          keep_indeces.append(index)
  
  return df.loc[keep_indeces],keep_indeces

#Return dataframe from set of keys with run info as index
def get_df(tree,keys):
  """Input tree from uproot and keys, return df with indeces
  If dfs are different sizes, we'll return a list of dfs
  """
  copy_keys = keys.copy() #Avoid modifying input
  copy_keys.extend(hdrkeys)
  df = tree.arrays(copy_keys,library='pd')
  if isinstance(df,tuple): #If it's a tuple, we'll rename each df, and return the list of them
    dfs = []
    for tempdf in df:
      tempdf = tempdf.set_index(hdrkeys)
      tempdf = tempdf.sort_index()
      dfs.append(tempdf)
    return dfs

  else: #Returns single df
    df = df.set_index(hdrkeys)
    df = df.sort_index()
    return df
    
def cut_pdg_event(df,pdg,pdg_key=f'{primprefix}pdg',status_key=f'{primprefix}gstatus',T_key=f'{primprefix}genT',E_threshold=0,exclude=True,
                max_count=1):
  
  """Exclude true to remove events with pdg, otherwise we'll include"""
  #Max count is max number of events allowed
  df = df.sort_index()
  drop_inds = []
  indeces = list(df.index.drop_duplicates()) #Get run info for each event
  start = time()
  for ind in indeces:
    row = df.loc[ind]
    row = row[row.loc[:,status_key] == 1] #Check outgoing final state particles
    row = row[abs(row.loc[:,pdg_key]) == pdg] #Keep only correct pdg in event
    Ts = row[T_key]
    if len([x for x in Ts if x >= E_threshold]) >= max_count: #An event exceeding threshold E
      drop_inds.append(ind)
  if not exclude: #Switch this to inds we want to keep
    df = df.loc[drop_inds]
  else:
    df = df.drop(drop_inds)
  return df,drop_inds

def cut_nshws(df,nshw,nshw_key=f'{recoprefix}nshw'):
  """Drop events that don't match nshws specified"""
  keep_inds = []
  indeces = list(df.index.drop_duplicates()) #Get run info for each event
  start = time()
  for ind in indeces:
    if df.loc[ind,nshw_key] == nshw: #Number of showers equals target number
      keep_inds.append(ind)
  return df.loc[keep_inds],keep_inds

def cut_ntrks(df,ntrk,ntrk_key=f'{recoprefix}ntrk'):
  """Drop events that don't match ntrks specified"""
  keep_inds = []
  indeces = list(df.index.drop_duplicates()) #Get run info for each event
  start = time()
  for ind in indeces:
    if df.loc[ind,ntrk_key] == ntrk: #Number of showers equals target number
      keep_inds.append(ind)
  return df.loc[keep_inds],keep_inds

def true_nue(df,pdg_key=f'{primprefix}pdg',drop=True):
  """Record events that are truth nu e scattering"""
  drop_inds = []
  indeces = list(df.index.drop_duplicates()) #Get run info for each event
  start = time()
  for ind in indeces:
    es = 0 #Count electrons
    nus = 0 #Count neutrinos
    others = 0 #Count anything else
    pdgs = df.loc[ind,pdg_key]
    for pdg in pdgs:
      if abs(pdg) == 11:
        es+=1
      elif abs(pdg) == 12 or abs(pdg) == 14:
        nus += 1
      else:
        others+=1
    if es == 1 and nus == 1 and others == 0: #this is nu e scattering
      drop_inds.append(ind)
  return df.drop(drop_inds),drop_inds
      
def number_events(df):
    """Return number of events in a dataframe"""
    return df.index.drop_duplicates().shape[0]

def get_shw_confusion_matrix(recodf,mcdf,nshw_key=f'{recoprefix}nshw',pdg_key=f'{primprefix}pdg',n=20,normalize=True):
  """Return matrix with number of showers on y, reco showers on x axis
  
  nshw_key: key in caf counting showers
  n: number of particles to make nxn matrix, 20 by default"""
  indeces_reco = list(recodf.index.drop_duplicates()) #Get run info for each event
  indeces_mc = list(mcdf.index.drop_duplicates()) #Get run info for each event
  start = time()
  conmat = np.zeros((n,n))
  #common_indeces = list(set(indeces_reco).intersection(indeces_mc))
  for ind in indeces_mc:
    mcpdgs = abs(mcdf.loc[ind,pdg_key]) #mc pdgs
    trueshws = 0 #Count number of showers
    for pdg in mcpdgs:
      if abs(pdg) == 11: #e
        trueshws += 1
      elif abs(pdg) == 22: #photon
        trueshws += 1
      elif abs(pdg) == 111: #pi0
        trueshws += 2 #Assume it makes two photons -> two showers
    if ind not in indeces_reco: #Keep only events in common, store non matches as other
      recoshws = 0
      #print(ind)
    else:
      recoshws = recodf.loc[ind,nshw_key]
    conmat[recoshws,trueshws] += 1
  
  return conmat

def get_trk_confusion_matrix(recodf,mcdf,ntrk_key=f'{recoprefix}ntrk',pdg_key=f'{primprefix}pdg',n=20):
  """Return matrix with number of showers on y, reco showers on x axis
  
  ntrk_key: key in caf counting showers
  n: number of particles to make nxn matrix, 20 by default"""
  indeces_reco = list(recodf.index.drop_duplicates()) #Get run info for each event
  indeces_mc = list(mcdf.index.drop_duplicates()) #Get run info for each event
  start = time()
  conmat = np.zeros((n,n))
  trkpdgs = [2212,211,13] #Additional tracks :[3112,321,4122,3222]
  for ind in indeces_mc:
    #if ind not in indeces_reco: #Keep only events in common, store non matches as other
    #  conmat
    mcpdgs = list(abs(mcdf.loc[ind,pdg_key])) #mc pdgs
    truetrks = 0 #Count number of showers
    for trkpdg in trkpdgs:
      truetrks += mcpdgs.count(trkpdg)
    if ind not in indeces_reco: #Keep only events in common, store non matches as other
      recotrks = 0
    else:
      recotrks = recodf.loc[ind,ntrk_key]
    conmat[recotrks,truetrks] += 1
  return conmat

def cut_razzlescore(df,score_key=f'{shwprefix}razzle.electronScore',cutoff=0.5,nshw=1):
  """Return dataframe with scores above cutoff razzle score, works for nshw shower"""
  indeces = df.index.drop_duplicates()
  pass_inds = []
  for ind in indeces:
    #print(df.loc[ind,score_key])
    scores = [df.loc[ind,score_key]]
    if len([score for score in scores if score>cutoff]) == nshw: #True if there are nshw scores greater than cutoff
      pass_inds.append(ind)
  return df.loc[pass_inds]

def find_index_with_key(dfs,key):
  """Find df which contains key specified, returns indeces of df list
  It should only return one index"""
  indeces = []
  if isinstance(dfs,list):
    for i,df in enumerate(dfs):
      keys = df.keys()
      if key in keys:
        #return i
        indeces.append(i)
    #print(indeces)
    if len(indeces) == 1:
      return indeces[0]
    else:
      Warning('This key shows up in multiple dataframes.')
    return indeces
  else:
    return None #This means there's only one dataframe, and we don't need to index a list

def FV_cut(df,x_key=f'{mcnuprefix}position.x',y_key=f'{mcnuprefix}position.y',z_key=f'{mcnuprefix}position.z',
  xmax=175,xmin=1.5,ymax=175,zmin=20,zmax=470):
  """Make truth level fidicual cut, we won't see these events"""
  keep_indeces = []
  indeces = df.index.drop_duplicates()
  for ind in indeces:
    #Get genie vertex coords.
    x = df.loc[ind,x_key]
    y = df.loc[ind,y_key]
    z = df.loc[ind,z_key]

    if abs(x) > xmin and abs(x) < xmax and abs(y) < ymax and abs(z) > zmin and abs(z) < zmax:
      keep_indeces.append(ind)
  return keep_indeces #Multiindex to index df

def cut_recoE(df,E_key=f'{shwprefix}bestplane_energy',cutoff=0.2,npass=1):
  """Cut events using reco energy > cutoff
  npass: number of reco events that must have greater than this energy
  """
  keep_indeces = []
  indeces = df.index.drop_duplicates()
  for ind in indeces:
    Es = df.loc[ind,E_key]
    if isinstance(Es,np.float64) or isinstance(Es,np.float32) and npass == 1: #Handle single value
      if Es >= cutoff: #Passes cutoff
        keep_indeces.append(ind)
    else:
      if len([x for x in Es if x >= cutoff]) >= npass: #An event exceeding threshold E
        keep_indeces.append(ind)
  return keep_indeces #Multiindex to index df




    
  


