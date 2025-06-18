#!/usr/bin/env python
# -*- coding: utf-8 -*-

# File-Name:    dimsSpikeCrossDimsProjI.py
# Version:      0.0.3 (see end of script for version descriptions)

############ Setup ############
# Import packages
import sys
import pandas as pd
pd.options.mode.chained_assignment = None # Turn off warnings for chained assignments
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib
from itertools import product
import itertools
from sklearn.utils import resample
from scipy.spatial import distance
import copy

# Import own packages
sys.path.append('...')
from Functions.Import.LocationsI import spikesDataII, resultsFolderI
from Functions.Preprocessing.dictionariesI import arealDicII
from PreAnalysis.fsSpike_GallC_100ms_StimOn_Go_II import anchortimes, import_processed_spike_data, create_time_window_names
from Functions.Preprocessing.MazeMetadataI import mazes

# Import results from prior analyses
from Dimensions.dimsSpikeGoalI_2ndPopulation import simple_projection_routine as goalI_projection_routine
from Dimensions.dimsSpikeMoveI_2ndPopulation import simple_projection_routine as moveI_projection_routine
from Dimensions.dimsSpikeGoalI_2ndPopulation import areas_sweep, create_data_subselection, parse_args

# Create a copy of projection routine function and set the pca_components parameter to 4
def modified_projection_routine(pca_components=4, is_goal=True):
    def wrapper(*args, **kwargs):
        # Import correct module based on type
        if is_goal:
            import Dimensions.dimsSpikeGoalI_2ndPopulation as mod
            original_func = goalI_projection_routine
        else:
            import Dimensions.dimsSpikeMoveI_2ndPopulation as mod
            original_func = moveI_projection_routine
            
        original_components = mod.pca_components
        mod.pca_components = pca_components
        result = original_func(*args, **kwargs)
        mod.pca_components = original_components
        return result
    return wrapper
goalI_projection_routine_pca4 = modified_projection_routine(4, is_goal=True)
moveI_projection_routine_pca4 = modified_projection_routine(4, is_goal=False)

pca_spaces_dict = {
    'goalc24':goalI_projection_routine_pca4,
    'movec24':moveI_projection_routine_pca4
}

# Locations
resultsLoc = resultsFolderI+"Dimensions/dimsSpikeCrossDimsProjI/"

# Import data
analysisdf_import = import_processed_spike_data()

# Create dicts to translate Npath to move direction
## For choice 1
NPath2Move_C1 = mazes.loc[(mazes['ChoiceNo']=='ChoiceI')][['Npath','NextFPmap']].set_index('Npath').to_dict()['NextFPmap']
analysisdf_import['Move_c1'] = analysisdf_import['PathNum'].map(NPath2Move_C1)
## For choice 2
NPath2Move_C2 = mazes.loc[(mazes['ChoiceNo']=='ChoiceII')][['Npath','NextFPmap']].set_index('Npath').to_dict()['NextFPmap']
analysisdf_import['Move_c2'] = analysisdf_import['PathNum'].map(NPath2Move_C2)
## For choice 3
NPath2Move_C3 = mazes.loc[(mazes['ChoiceNo']=='ChoiceIII')][['Npath','NextFPmap']].set_index('Npath').to_dict()['NextFPmap']
analysisdf_import['Move_c3'] = analysisdf_import['PathNum'].map(NPath2Move_C3)
## For choice 4
NPath2Move_C4 = mazes.loc[(mazes['ChoiceNo']=='ChoiceIV')][['Npath','NextFPmap']].set_index('Npath').to_dict()['NextFPmap']
analysisdf_import['Move_c4'] = analysisdf_import['PathNum'].map(NPath2Move_C4)

# Window data for projection
goal_locations = [7,9,17,19]
move_directions = ['u','l','d','r']

# Project each path's mean activity from each window onto each dimension
pathvar_options = list(product(goal_locations,move_directions))
pathvar_options = ['_'.join([str(i) for i in i_path]) for i_path in pathvar_options]

# Figure settings
# Create font information 
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 6,6,7
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

## Get 4 RGB codes from viridis colormap for goals
colours = plt.cm.viridis(np.linspace(0,1,len(goal_locations)))
goal_colours = dict(zip(goal_locations,colours))

## Define 4 markers for goals
move_markers = {
    'l':'<',
    'r':'>',
    'u':'^',
    'd':'v'
}

############ Functions ############

def create_pca_dicts(curr_area_sweep, pca_spaces_dict):
    '''
    Import PCAs from other scripts
    '''
    # Get the pca dimensions
    pca_dict = {}
    pca_data_dicts = {}
    for i_key, i_value in pca_spaces_dict.items():
        _, pca, _, data_df = i_value(curr_area_sweep, save_figure =False, return_df = True)
        pca_dict[i_key] = pca
        pca_data_dicts[i_key] = data_df

    return pca_dict, pca_data_dicts

def pca_var_explained_plot(pca_dict, curr_area_sweep):
    '''
    Create plot of variance explained by each PCA dimension.
    '''
    # Create figure
    fig, ax = plt.subplots()
    for i_key, i_value in pca_dict.items():
        ex_vars = i_value.explained_variance_ratio_
        df = pd.DataFrame({
            'explained_variance': ex_vars,
            'cumulative_explained_variance': np.cumsum(ex_vars)
        })
        df.to_csv(f"{resultsLoc}pca_variance_{i_key}_{curr_area_sweep[1]}.csv")
        ax.plot(ex_vars, label=i_key)
    ax.legend()
    ax.set_title('Variance explained by PCA dimensions, {}'.format(curr_area_sweep[1]))
    ax.set_xlabel('PCA dimension')
    ax.set_ylabel('Variance explained')
    plt.savefig(resultsLoc+'PCA_variance_explained_{}.png'.format(curr_area_sweep[1]),dpi = 300)
    plt.close()

############ Execution calls ############

for i_area_sweep in areas_sweep:
    # Get pca dicts
    pca_dict, pca_data_dicts = create_pca_dicts(i_area_sweep, pca_spaces_dict)

    # Plot variance explained
    pca_var_explained_plot(pca_dict, i_area_sweep)

    # Make figure for projection movec24 data into goalc24 space
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    for i_move in move_directions:
        data = pca_data_dicts['movec24'].reset_index()
        i_data = data[data['Move']==i_move]['dim_window'].values.reshape(1,-1)
        proj_vec = pca_dict['goalc24'].transform(i_data).squeeze(0)[:2]
        ax.scatter(proj_vec[0],proj_vec[1],label=i_move,marker=move_markers[i_move],color='black')
    for i_goal in goal_locations:
        data = pca_data_dicts['goalc24'].reset_index()
        i_data = data[data['FinalPos']==i_goal]['dim_window'].values.reshape(1,-1)
        proj_vec = pca_dict['goalc24'].transform(i_data).squeeze(0)[:2]
        ax.scatter(proj_vec[0],proj_vec[1],label=i_goal,marker='o',color=goal_colours[i_goal])
    ax.legend()
    ax.set_title('Projection into goal space. {}'.format(i_area_sweep[1]))
    ax.set_xlabel('PCA dimension 1')
    ax.set_ylabel('PCA dimension 2')
    plt.tight_layout()
    plt.savefig(resultsLoc+'Projection_into_goalc24_{}.png'.format(i_area_sweep[1]),dpi = 300)
    plt.close()

    # Make figure for projection goalc24 data into movec24 space
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    for i_goal in goal_locations:
        data = pca_data_dicts['goalc24'].reset_index()
        i_data = data[data['FinalPos']==i_goal]['dim_window'].values.reshape(1,-1)
        proj_vec = pca_dict['movec24'].transform(i_data).squeeze(0)[:2]
        ax.scatter(proj_vec[0],proj_vec[1],label=i_goal,marker='o',color=goal_colours[i_goal])
    for i_move in move_directions:
        data = pca_data_dicts['movec24'].reset_index()
        i_data = data[data['Move']==i_move]['dim_window'].values.reshape(1,-1)
        proj_vec = pca_dict['movec24'].transform(i_data).squeeze(0)[:2]
        ax.scatter(proj_vec[0],proj_vec[1],label=i_move,marker=move_markers[i_move],color='black')
    ax.legend()
    ax.set_title('Projection into move space. {}'.format(i_area_sweep[1]))
    ax.set_xlabel('PCA dimension 1')
    ax.set_ylabel('PCA dimension 2')
    plt.tight_layout()
    plt.savefig(resultsLoc+'Projection_into_movec24_{}.png'.format(i_area_sweep[1]),dpi = 300)
    plt.close()

    # Run cross validated cross projection analysis
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import explained_variance_score
    from Dimensions.dimsSpikeGoalI_2ndPopulation import create_GoalDimensions_C2C4, get_multiindex_goal, create_data_subselection
    from Dimensions.dimsSpikeGoalI_2ndPopulation import windows_for_dims_dicts as windows_for_dims_dicts_goalspace
    from Dimensions.dimsSpikeMoveI_2ndPopulation import create_MoveDimensions_C2C4, get_multiindex_move
    from Dimensions.dimsSpikeMoveI_2ndPopulation import windows_for_dims_dicts as windows_for_dims_dicts_movespace
    from Dimensions.dimsSpikeGoalI_2ndPopulation import areas_sweep, create_data_subselection, parse_args
    dim_names = ['Goal','Move']

    # Create data subselection
    analysisdf, unit_list, data_area = create_data_subselection(i_area_sweep, analysisdf_import)
    analysisdf.reset_index(inplace=True, drop=True)
    unit_list = analysisdf['Unit'].unique()

    # Split analysisdf into two halves, using sklearn and stratified by unit, goal
    analysisdf['statification_var'] = analysisdf['Unit'].astype(str)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    analysisdf_halves = []
    for train_index, test_index in splitter.split(analysisdf, analysisdf['statification_var']):
        analysisdf_halves.append(analysisdf.iloc[train_index])
        analysisdf_halves.append(analysisdf.iloc[test_index])

    # Create PCAs and training datas
    pca_dict = {}
    pca_data = {}
    reconstruction_dfs = {}
    for i_num, half_df in enumerate(analysisdf_halves):
    
        # For GoalI
        pca_dict['Goal_'+str(i_num)], pca_data['Goal_'+str(i_num)] = create_GoalDimensions_C2C4(half_df, windows_for_dims_dicts_goalspace, unit_list = unit_list, pca_components=2)
        # For MoveI
        pca_dict['Move_'+str(i_num)], pca_data['Move_'+str(i_num)] = create_MoveDimensions_C2C4(half_df, windows_for_dims_dicts_movespace, unit_list = unit_list, pca_components=2)

        # Create dataframes to hold reconstruction data
        reconstruction_dfs['PCA_'+str(i_num)] = pd.DataFrame(index=dim_names, columns=dim_names)

    # Reconstruct data across two halves
    directions = [(0,1),(1,0)]
    for pca_half, data_half in directions:
        for pca_key, data_key in list(product(dim_names, dim_names)):
            # Get model and data
            curr_proj_data = pca_data[data_key+'_'+str(data_half)]
            curr_pca_model = pca_dict[pca_key+'_'+str(pca_half)]
            # Project
            proj_data = curr_pca_model.transform(curr_proj_data)
            inv_proj_data = curr_pca_model.inverse_transform(proj_data)
            # Calculate explained variance
            varexp = explained_variance_score(curr_proj_data,inv_proj_data, multioutput='variance_weighted')
            # Store
            reconstruction_dfs['PCA_'+str(pca_half)].loc[pca_key,data_key] = varexp

    # Take elementwise mean of both dfs in reconstruction_dfs
    combined_df = (reconstruction_dfs['PCA_0']+reconstruction_dfs['PCA_1']) / 2
    combined_df.columns = [f"{col}_data" for col in combined_df.columns]
    combined_df.index = [f"{idx}_pca" for idx in combined_df.index]

    # Save to disk 
    combined_df.to_csv(resultsLoc+'ReconstructionVarExplained_{}.csv'.format(i_area_sweep[1]))
    del pca_dict, pca_data_dicts, combined_df, reconstruction_dfs
