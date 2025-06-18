#!/usr/bin/env python
# -*- coding: utf-8 -*-

# File-Name:    MazeMetadataI.py
# Version:      1.0.1 (see end of script for version descriptions)

# Import packages
from typing import BinaryIO
import pandas as pd
import numpy as np
import os

# Import File
from Functions.Import.LocationsI import problemDataI 
mazesFrame = pd.read_csv(problemDataI+'PathsFullNumbered.csv', sep=';')
mazesFrame.rename(columns={'Unnamed: 0':'Type'}, inplace=True)

# Create frame used for transformations
df = mazesFrame

# Add a counter for each state (used by e.g. Model Free Learning Algorithm)
df.reset_index(inplace=True, drop=True)
df['StateNumber']=df.index

# Add column with middle point of current decision
df['ALT']=df['ALT'].shift(-1)
df['BI']=df['BI'].shift(-1)
df['BII']=df['BII'].shift(-1)
df['NextFP']=df['FP'].shift(-1)

# Set coordinates relative to middle point of decision
## Calculate numeric difference of coordinate number to middle point
df = df.assign(
    FPrel=df['FP']-df['NextFP'],
    ALTrel=df['FP']-df['ALT'],
    BIrel=df['FP']-df['BI'],
    BIIrel=df['FP']-df['BII']
)
## Map that difference to the relative direction to middle point
directionDic ={1:'l', -1:'r',5:'u',-5:'d'}
df = df.assign(
    NextFPmap=df['FPrel'].map(directionDic),
    ALTmap=df['ALTrel'].map(directionDic)
)
## Map the next FP to numeric action value
directionNumeric ={'l':1,'u':2,'r':3,'d':4}
df['NextFPnum'] = df['NextFPmap'].map(directionNumeric)

## Create column with the second next step - note that this variable is transformed later in the script to represent the direction planned under 2 step solution during the first choice
df['NextFPIImap'] = df['NextFPmap'].shift(-1)

# Add columns per relative direction
directions = ['l','r','u','d']
for field_type in ['NextFP', 'ALT']:
  for direction in directions:
    df.loc[df[field_type+'map']==direction,direction]=field_type
df.fillna({'l':'Block','r':'Block','u':'Block','d':'Block'}, inplace=True)

# Add column with number of steps per maze
df['Nsteps']=df.groupby('Npath').cumcount()
df['Nsteps']=df.groupby('Npath')['Nsteps'].transform('max')

# Extract vector with index numbers where new mazes start
mazeIDs = df.loc[df['FP']==13].index.values

## Add column with pattern of two available choice pattern. The order of letters gets "normalised" to get a key for choice patterns that is independent of FP / ALT direction.
df['ChoicesIdent']=df['NextFPmap']+df['ALTmap']
anonymiseChoiceDic = {'rl':'lr','du':'ud','lu':'ul','dl':'ld','ru':'ur','dr':'rd', # Change mapping for one half
                      'lr':'lr','ud':'ud','ul':'ul','ld':'ld','ur':'ur','rd':'rd'
                      }
df['ChoicesCategory']=df['ChoicesIdent'].map(anonymiseChoiceDic)

#Save results in dataframe accessed by external scripts
mazes = df

# Add goal information to dataframe
## Single coordinate
def extract_endgoal(group):
    #Accesses the FP of the last row of a given group
    group['goal']=group.iloc[-1]['FP']

    return group
mazes=mazes.groupby('Npath').apply(extract_endgoal)

## Hemispheric
dic_x = {7:1,17:1,9:2,19:2}
dic_y = {7:2,17:1,9:2,19:1}
mazes['goal_x']=mazes['goal'].map(dic_x)
mazes['goal_y']=mazes['goal'].map(dic_y)

# Create column with ChoiceNo variable as transformation of type
typeDic ={'start':'ChoiceI','step 1':'ChoiceII','step 2':'ChoiceIII','step 3':'ChoiceIV', 'step 4':'Goal'}
mazes['ChoiceNo'] = mazes['Type'].map(typeDic)
mazes.loc[(mazes['Nsteps']==2)&(mazes['ChoiceNo']=='ChoiceIII'),'ChoiceNo']='Goal'

# Add columns specifying what field a given block is on
## Types of fields
'''
During first choice:
- close: blocks field in goal hemisphere
- far: blocks field outside goal hemisphere
During second choice:
- far: blocks way back (13; prior FP)
- target: blocks goal location (4step scenario)
- long: blocks 4 step route (2 step scenario)
'''
mazes['BIident'] = np.nan
mazes['BIIident'] = np.nan

## Choice 1:
### By checking whether a blocks location number is in [1,-1,5,-5] one can check whether it is in goals hemisphere
for i in np.arange(1,25):
  currGoal = mazes.loc[mazes['Npath']==i]['goal'].iloc[0]
  Bone = mazes.loc[(mazes['Npath']==i)&(mazes['ChoiceNo']=='ChoiceI')]['BI']
  Btwo = mazes.loc[(mazes['Npath']==i)&(mazes['ChoiceNo']=='ChoiceI')]['BII']
  distances = np.array([1,-1,5,-5])
  distances = distances + currGoal
  if np.isin(Bone, distances):
    mazes.loc[(mazes['Npath']==i)&(mazes['ChoiceNo']=='ChoiceI'),'BIident']='close'
    mazes.loc[(mazes['Npath']==i)&(mazes['ChoiceNo']=='ChoiceI'),'BIIident']='far'
  elif np.isin(Btwo, distances):
    mazes.loc[(mazes['Npath']==i)&(mazes['ChoiceNo']=='ChoiceI'),'BIident']='far'
    mazes.loc[(mazes['Npath']==i)&(mazes['ChoiceNo']=='ChoiceI'),'BIIident']='close'

## Choice 2:
### One of the blocks is on past FP (=13) and the other one depend on 2 step or 4 step process
for i in np.arange(1,25):
  currSteps= mazes.loc[mazes['Npath']==i]['Nsteps'].iloc[0]
  if currSteps == 2:
    altBlock = 'long'
  if currSteps == 4:
    altBlock = 'target'
  Bone = mazes.loc[(mazes['Npath']==i)&(mazes['ChoiceNo']=='ChoiceII')]['BI']
  Btwo = mazes.loc[(mazes['Npath']==i)&(mazes['ChoiceNo']=='ChoiceII')]['BII']
  if Bone.values==13:
    mazes.loc[(mazes['Npath']==i)&(mazes['ChoiceNo']=='ChoiceII'),'BIident']='far'
    mazes.loc[(mazes['Npath']==i)&(mazes['ChoiceNo']=='ChoiceII'),'BIIident']=altBlock
  elif Btwo.values==13:
    mazes.loc[(mazes['Npath']==i)&(mazes['ChoiceNo']=='ChoiceII'),'BIident']=altBlock
    mazes.loc[(mazes['Npath']==i)&(mazes['ChoiceNo']=='ChoiceII'),'BIIident']='far'

## Choice 3 & 4: Both blocks are blocking past FP during Choice 3 & 4 in 4 Step mazes
mazes.loc[mazes['ChoiceNo'].isin(['ChoiceIII','ChoiceIV']), ['BIident', 'BIIident']]='far'


### Transform NextFPIImap variable to represent the choice planned under 2 step solution during first step
for g in [7,9,17,19]:
  NextFPlist = mazes.loc[(mazes['Nsteps']==4)&(mazes['Type']=='start')&(mazes['goal']==g)]['NextFP'].unique()
  for i in NextFPlist:
    # Extract which choice the monkey would take under the corresponding 2 step maze
    plannedChoice = mazes.loc[(mazes['Nsteps']==2)&(mazes['Type']=='step 1')&(mazes['goal']==g)&(mazes['FP']==i)]['NextFPmap'].values
    # Set this choice for NextFPIImap
    mazes.loc[(mazes['Nsteps']==4)&(mazes['Type']=='start')&(mazes['goal']==g)&(mazes['NextFP']==i), 'NextFPIImap'] = plannedChoice[0]
