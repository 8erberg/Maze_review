#!/usr/bin/env python
# -*- coding: utf-8 -*-

# File-Name:    dimsSpikeGoalI_2ndPopulation.py
# Version:      0.0.6

############ Setup ############
# Import packages
import sys
import pandas as pd
pd.options.mode.chained_assignment = None # Turn off warnings for chained assignments
import numpy as np
np.random.seed(23)
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import explained_variance_score
from statsmodels.stats.multitest import fdrcorrection
from itertools import product
from scipy import stats

# Import own packages
sys.path.append('...')
from Functions.Import.LocationsI import spikesDataII, resultsFolderI
from Functions.Preprocessing.dictionariesI import arealDicII
from PreAnalysis.fsSpike_GallC_100ms_StimOn_Go_II import anchortimes, import_processed_spike_data, create_time_window_names
from Functions.Preprocessing.MazeMetadataI import mazes

# Locations
resultsLoc = resultsFolderI+"Dimensions/dimsSpikeGoalI_2ndPopulation/"

# Execution variables
areas_sweep = [
    (['dPFC'],'dPFC'),(['dmPFC'],'dmPFC'),(['vPFC'],'vPFC'),(['Insula'],'Insula'),(['dlPM'],'dlPM'),
    (['PS'],'PS'),(['dPFC', 'dmPFC','dlPM','PS','Insula','vPFC'],'all')]
#windows_for_dims_dicts = {
#    'C2':['UNFIXTG21_ON_-0.3','UNFIXTG21_ON_-0.2','UNFIXTG21_ON_-0.1', 'UNFIXTG21_ON_0.0'],
#    'C4':['UNFIXTG41_ON_-0.3','UNFIXTG41_ON_-0.2','UNFIXTG41_ON_-0.1', 'UNFIXTG41_ON_0.0']}
windows_for_dims_dicts = {
    'C2':['UNFIXTG21_ON_-0.2','UNFIXTG21_ON_-0.1', 'UNFIXTG21_ON_0.0'],
    'C4':['UNFIXTG41_ON_-0.2','UNFIXTG41_ON_-0.1', 'UNFIXTG41_ON_0.0']}
grouped_projection_windows = {
        'EarlyGoal':('1_all',['TRGDLY_0.1','TRGDLY_0.2','TRGDLY_0.3']),
        'LateGoal':('1_all',['TRGDLY_0.4','TRGDLY_0.5','TRGDLY_0.6', 'TRGDLY_0.7','TRGDLY_0.8']),
        'EarlyC1':('1_all',['UNFIXTG11_ON_-0.2','UNFIXTG11_ON_-0.1', 'UNFIXTG11_ON_0.0']),
        'LateC1':('1_all',['UNFIXTG11_ON_0.2', 'UNFIXTG11_ON_0.3']),
        'GoC1':('1_all',['FIXTG0_OFF_-0.2','FIXTG0_OFF_-0.1', 'FIXTG0_OFF_0.0']),
        'EarlyC2_2step':('2_2step',['UNFIXTG21_ON_-0.2', 'UNFIXTG21_ON_-0.1', 'UNFIXTG21_ON_0.0']),
        'LateC2_2step':('2_2step',['UNFIXTG21_ON_0.2', 'UNFIXTG21_ON_0.3']),
        'GoC2_2step':('2_2step',['FIXTG11_OFF_-0.2', 'FIXTG11_OFF_-0.1', 'FIXTG11_OFF_0.0']),
        'EarlyC2_4step':('2_4step',['UNFIXTG21_ON_-0.2', 'UNFIXTG21_ON_-0.1', 'UNFIXTG21_ON_0.0']),
        'LateC2_4step':('2_4step',['UNFIXTG21_ON_0.2', 'UNFIXTG21_ON_0.3']),
        'GoC2_4step':('2_4step',['FIXTG11_OFF_-0.2', 'FIXTG11_OFF_-0.1', 'FIXTG11_OFF_0.0']),
        'EarlyC3_4step':('3_4step',['UNFIXTG31_ON_-0.2', 'UNFIXTG31_ON_-0.1', 'UNFIXTG31_ON_0.0']),
        'LateC3_4step':('3_4step',['UNFIXTG31_ON_0.2', 'UNFIXTG31_ON_0.3']),
        'GoC3_4step':('3_4step',['FIXTG21_OFF_-0.2', 'FIXTG21_OFF_-0.1', 'FIXTG21_OFF_0.0']),
        'EarlyC4_4step':('4_4step',['UNFIXTG41_ON_-0.2', 'UNFIXTG41_ON_-0.1', 'UNFIXTG41_ON_0.0']),
        'LateC4_4step':('4_4step',['UNFIXTG41_ON_0.2', 'UNFIXTG41_ON_0.3']),
        'GoC4_4step':('4_4step',['FIXTG31_OFF_-0.2', 'FIXTG31_OFF_-0.1', 'FIXTG31_OFF_0.0'])}
grouped_projection_windows_cleanedNames = {
        'EarlyC1':'C1, early',
        'LateC1':'C1, late',
        'GoC1':'C1, go',
        'EarlyC2_2step':'C2 (goal), early',
        'LateC2_2step':'C2 (goal), late',
        'GoC2_2step':'C2 (goal), go',
        'EarlyC2_4step':'C2 (away), early',
        'LateC2_4step':'C2 (away), late',
        'GoC2_4step':'C2 (away), go',
        'EarlyC3_4step':'C3, early',
        'LateC3_4step':'C3, late',
        'GoC3_4step':'C3, go',
        'EarlyC4_4step':'C4, early',
        'LateC4_4step':'C4, late',
        'GoC4_4step':'C4, go'}
windows_for_projection_series = pd.Series(grouped_projection_windows.keys())
windows_for_projection = list(grouped_projection_windows.keys())
x_vals = np.array(range(len(windows_for_projection)))
goal_numbers = [7,9,17,19]
pca_components = 3

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

############ Functions ############

def shuffle_goal(group):
    '''
    Shuffle gaol columns within grouped df.
    '''
    group['FinalPos'] = np.random.permutation(group['FinalPos'])
    return group

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-area_num", type=int, 
                        help="Area for analysis, chosen as numerical from list")
    args = parser.parse_args()
    area_num = args.area_num
    return area_num

def create_path_num_dict(mazes):
    '''
    Create a dictionary of path numbers for each path length
    '''
    path_nums_all = mazes.Npath.unique()
    path_nums_2step = mazes.loc[(mazes['Nsteps']==2)].Npath.unique()
    path_nums_4step = mazes.loc[(mazes['Nsteps']==4)].Npath.unique()
    path_nums_dict = {'all': path_nums_all, '2step':path_nums_2step, '4step':path_nums_4step}
    return path_nums_dict

def create_data_subselection(i_areas_sweep, analysisdf_import):
    '''
    Select area specific data.
    '''
    # Make copy of data for current run
    analysisdf = analysisdf_import.copy()

    # Select data based on area
    analysisdf['area'] = analysisdf['rec_area'].map(arealDicII)
    analysisdf = analysisdf.loc[analysisdf['area'].isin(i_areas_sweep[0])]
    analysisdf['Unit'] = analysisdf['session']+'_'+analysisdf['NeuronNum'].astype(str)
    data_area = i_areas_sweep[1]

    # Create list of units
    unit_list = analysisdf['Unit'].unique()

    return analysisdf, unit_list, data_area

def create_GoalDimensions_C2C4(analysisdf,windows_for_dims_dicts, unit_list, pca_components=3, return_df=False):
    '''
    Train Goal PCA Dimensions on C2 and C4 windows using windows_for_dims_dicts.
    '''

    PCA_data_list = []
    path_nums_dict = create_path_num_dict(mazes)

    # Get multiindex
    multi_index = get_multiindex_goal(unit_list)

    # Get mean activation for choice 2
    windows_for_dims = windows_for_dims_dicts['C2']
    temp_df = analysisdf.copy()
    temp_df = temp_df.loc[(temp_df['Choice2step']==1)].dropna(subset=['FIXTG11_OFF_0.0'])
    temp_df = temp_df.loc[temp_df['PathNum'].isin(path_nums_dict['2step'])]
    temp_df['dim_window'] = temp_df[windows_for_dims].mean(axis=1)
    temp_df = temp_df.groupby(['Unit','FinalPos']).mean()['dim_window']
    temp_df = temp_df.reindex(multi_index).reset_index()
    temp_df['ChoiceInfo']='C2'
    PCA_data_list.append(temp_df)

    # Get mean activation for choice 4
    windows_for_dims = windows_for_dims_dicts['C4']
    temp_df = analysisdf.copy()
    temp_df = temp_df.loc[(temp_df['Choice4step']==1)].dropna(subset=['FIXTG31_OFF_0.0'])
    temp_df['dim_window'] = temp_df[windows_for_dims].mean(axis=1)
    temp_df = temp_df.groupby(['Unit','FinalPos']).mean()['dim_window']
    temp_df = temp_df.reindex(multi_index).reset_index()
    temp_df['ChoiceInfo']='C4'
    PCA_data_list.append(temp_df)

    # Concatenate dataframes
    pca_analysisdf = pd.concat(PCA_data_list)

    # Calculate mean unit acitivity per unit per trial type
    ## Unweight mean for choice time point
    train_trial_mean_df = pca_analysisdf.groupby(['Unit','FinalPos']).mean()['dim_window']
    train_trial_mean_df = train_trial_mean_df.reindex(multi_index)
    train_trial_mean_df = train_trial_mean_df.unstack(level=0, fill_value=0) 
    X = train_trial_mean_df.values

    # Calculate PCA
    pca = PCA(n_components=pca_components)
    pca.fit(X)

    if return_df:
        return pca, pca_analysisdf.groupby(['Unit','FinalPos']).mean()['dim_window']
    else:
        return pca, X

def get_multiindex_goal(unit_list_input):
    '''
    Create multiindex used for groupby of data.
    '''
    unit_list = unit_list_input.copy()
    unit_list.sort()

    var_combinations = [7,9,17,19]
    var_combinations = list(product(unit_list,var_combinations))
    var_combinations = [(i[0],i[1]) for i in var_combinations]
    multi_index = pd.MultiIndex.from_tuples(var_combinations, names=['Unit', 'FinalPos'])
    
    return multi_index

def fill_grouped_data_by_multiindex(grouped_data, multi_index, windows_for_projection):
    '''
    Take grouped dataframe and multindex to create full rank df via mean operation.
    '''
    # Update index to full index
    grouped_data_full = grouped_data.reindex(multi_index)
    # Fill missing values with group mean per Unit
    means = grouped_data_full.groupby(['Unit'])[windows_for_projection].transform('mean')
    grouped_data_full[windows_for_projection] = grouped_data_full[windows_for_projection].fillna(means)

    return grouped_data_full

def create_projection_data(analysisdf, grouped_projection_windows, windows_for_projection, unit_list, projection_groups = ['Unit','FinalPos'], shuffle_group = False):
    '''
    Create data_dict with one df corresponding to the dfs in grouped_projection_windows.
    '''
    # Create necessary variables
    go_times_list = ['FIXTG0_OFF','FIXTG11_OFF','FIXTG21_OFF','FIXTG31_OFF']
    path_nums_dict = create_path_num_dict(mazes)

    # Get multiidex dict
    multi_index = get_multiindex_goal(unit_list)

    # Get mean activation for time periods
    for key, value in grouped_projection_windows.items():
        analysisdf[key] = analysisdf[value[1]].mean(axis=1)

    # Create data_dict with a key for each df in grouped_projection_windows
    data_dict = {}
    for value in grouped_projection_windows.values():
        data_dict[value[0]] = None

    ## First choice point
    grouping_data = analysisdf.copy()
    grouping_data = grouping_data.dropna(subset=[go_times_list[0]+'_-0.2'])
    ### Shuffle goal, if shuffle_group == True
    if shuffle_group:
        grouping_data = grouping_data.groupby('Unit').apply(shuffle_goal)
    ### Create grouped means
    grouping_data = grouping_data.groupby(projection_groups).mean()[windows_for_projection]
    ### Fill missing values with group mean per Unit
    grouping_data = fill_grouped_data_by_multiindex(grouping_data, multi_index, windows_for_projection)
    grouping_data = grouping_data.reset_index()
    data_dict['1_all']=grouping_data
    ## Second choice point
    grouping_data = analysisdf.copy()
    grouping_data = grouping_data.dropna(subset=[go_times_list[1]+'_-0.2'])
    for pathway in ['2step','4step']:
        ### Select path numbers for current pathway
        path_nums = path_nums_dict[pathway]
        ### Select data for current pathway
        grouping_data_sub = grouping_data.loc[
            (grouping_data['PathNum'].isin(path_nums))&
            (grouping_data['Choice2step']==1)]
        ### Shuffle goal direction, if shuffle_group == True
        if shuffle_group:
            grouping_data_sub = grouping_data_sub.groupby('Unit').apply(shuffle_goal)
        ### Create grouped means
        grouping_data_sub = grouping_data_sub.groupby(projection_groups).mean()[windows_for_projection]
        ### Fill missing values with group mean per Unit
        grouping_data_sub = fill_grouped_data_by_multiindex(grouping_data_sub, multi_index, windows_for_projection)
        grouping_data_sub = grouping_data_sub.reset_index()
        ### Add data to dict
        data_dict['2_'+pathway]=grouping_data_sub
    ## Third & fourth choice point
    grouping_data = analysisdf.copy()
    pathway = '4step'
    ### Select path numbers for current pathway
    path_nums = path_nums_dict[pathway]
    ### Loop over choice points
    for choice_num in [3,4]:
        ### Select data for current pathway
        grouping_data_sub = grouping_data.loc[
            (grouping_data['PathNum'].isin(path_nums))&
            (grouping_data['Choice{0}step'.format(str(choice_num))]==1)]
        ### Drop rows with missing values 
        if choice_num == 3:
            grouping_data_sub = grouping_data_sub.dropna(subset=[go_times_list[2]+'_-0.2'])
        elif choice_num == 4:
            grouping_data_sub = grouping_data_sub.dropna(subset=[go_times_list[3]+'_-0.2'])
        ### Shuffle goal direction, if shuffle_group == True
        if shuffle_group:
            grouping_data_sub = grouping_data_sub.groupby('Unit').apply(shuffle_goal)
        ### Create grouped means
        grouping_data_sub = grouping_data_sub.groupby(projection_groups).mean()[windows_for_projection]
        ### Fill missing values with group mean per Unit
        grouping_data_sub = fill_grouped_data_by_multiindex(grouping_data_sub, multi_index, windows_for_projection)
        grouping_data_sub = grouping_data_sub.reset_index()
        ### Add data to dict
        data_dict['{0}_{1}'.format(str(choice_num),pathway)]=grouping_data_sub

    return data_dict

def project_set_of_windows(data_dict, grouped_projection_windows, pca, goal_numbers):
    '''
    Project each window in grouped_projection_windows onto the PCA dimensions.
    '''
    ## Create dict to hold results
    num_pca_dims = pca.components_.shape[0]
    dim_names = ['dim'+str(i) for i in range(1,num_pca_dims+1)]
    results_dict = {}
    for i_goal in goal_numbers:
        for i_key in dim_names:
            results_dict[i_key+'_'+str(i_goal)]=[]

    ## Project onto dims
    for i_goal in goal_numbers:
        for window_name, window_params in grouped_projection_windows.items():
            test_goal_mean_df = data_dict[window_params[0]]
            temp_pop_vector = test_goal_mean_df.loc[test_goal_mean_df['FinalPos']==i_goal][window_name].values.reshape(1,-1)
            # Project
            proj_vec = pca.transform(temp_pop_vector).flatten()

            # Add results to list
            for i_num, i_key in enumerate(dim_names):
                results_dict[i_key+'_'+str(i_goal)].append(proj_vec[i_num])

    return results_dict


def create_grid_plot_overview(results_dict, plot_lines, plot_name, resultsLoc):
    # Subselect data
    plot_dims = ['dim1','dim2']

    # Plot assignment
    window_groups_plot = {
        'EarlyGoal':[0,0],
        'LateGoal':[0,1],
        'EarlyC1':[1,0],
        'LateC1':[1,1],
        'GoC1':[1,2],
        'EarlyC2_2step':[2,0],
        'LateC2_2step':[2,1],
        'GoC2_2step':[2,2],
        'EarlyC2_4step':[3,0],
        'LateC2_4step':[3,1],
        'GoC2_4step':[3,2],
        'EarlyC3_4step':[4,0],
        'LateC3_4step':[4,1],
        'GoC3_4step':[4,2],
        'EarlyC4_4step':[5,0],
        'LateC4_4step':[5,1],
        'GoC4_4step':[5,2]
        }
    windows_for_projection = list(window_groups_plot.keys())

    ## Get 4 RGB codes from viridis colormap
    colours = plt.cm.viridis(np.linspace(0,1,len(plot_lines)))
    line_colours = dict(zip(plot_lines,colours))

    # Setup figure
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 14))
    st = fig.suptitle(plot_name)

    # Get data per goal projection
    for i_goal in plot_lines:
        x_vals = np.array(results_dict['{}_{}'.format(plot_dims[0],str(i_goal))])
        y_vals = np.array(results_dict['{}_{}'.format(plot_dims[1],str(i_goal))])

        # Plot data per time period
        for window_group_num, window_group_item in enumerate(window_groups_plot.items()):
            window_group_name = window_group_item[0]
            window_group_plot_pos =window_group_item[1]

            axes[window_group_plot_pos[0],window_group_plot_pos[1]].scatter(
                x_vals[window_group_num], y_vals[window_group_num], color=line_colours[i_goal]
            )

    # Add legend
    for i_goal in plot_lines:
        axes[0,0].plot([], [],'-', marker='o', color=line_colours[i_goal], alpha=0.5, label=str(i_goal))
    axes[0,0].legend(loc='center right', bbox_to_anchor=(1.3, 0.5))

    # Set subplot information
    for window_group_num, window_group_item in enumerate(window_groups_plot.items()):
        window_group_name = window_group_item[0]
        window_group_plot_pos =window_group_item[1]
        axes[window_group_plot_pos[0],window_group_plot_pos[1]].set_xlabel(plot_dims[0])
        axes[window_group_plot_pos[0],window_group_plot_pos[1]].set_ylabel(plot_dims[1])

        # Add title to axes
        axes[window_group_plot_pos[0],window_group_plot_pos[1]].set_title(window_group_name)


    # Add the same x and y axis limits to all subplots, scale them dynamically to min and max across all subplots
    x_min = np.min([ax.get_xlim()[0] for ax in axes.flatten()])
    x_max = np.max([ax.get_xlim()[1] for ax in axes.flatten()])
    y_min = np.min([ax.get_ylim()[0] for ax in axes.flatten()])
    y_max = np.max([ax.get_ylim()[1] for ax in axes.flatten()])
    for ax in axes.flatten():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Save figure
    plt.tight_layout()
    plt.savefig(resultsLoc+'{0}.png'.format(plot_name),dpi = 300)
    plt.close()

def create_grid_plot_print(results_dict, plot_lines, plot_name, resultsLoc):
    # Subselect data
    plot_dims = ['dim1','dim2']

    # Plot assignment
    window_groups_plot = {
        'EarlyC1':[0,0],
        'LateC1':[0,1],
        'GoC1':[0,2],
        'EarlyC2_2step':[1,0],
        'LateC2_2step':[1,1],
        'GoC2_2step':[1,2],
        'EarlyC2_4step':[2,0],
        'LateC2_4step':[2,1],
        'GoC2_4step':[2,2],
        'EarlyC3_4step':[3,0],
        'LateC3_4step':[3,1],
        'GoC3_4step':[3,2],
        'EarlyC4_4step':[4,0],
        'LateC4_4step':[4,1],
        'GoC4_4step':[4,2]
        }
    windows_for_projection = list(window_groups_plot.keys())

    ## Get 4 RGB codes from viridis colormap
    colours = plt.cm.viridis(np.linspace(0,1,len(plot_lines)))
    line_colours = dict(zip(plot_lines,colours))

    # Setup figure
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(2.3, 3.0))
    st = fig.suptitle(plot_name)

    SMALL_SIZE = 6
    MEDIUM_SIZE = 6
    BIGGER_SIZE = 7

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Adjust general plot parameters
    curr_linewidth = 0.5
    plt.subplots_adjust(hspace=0.3)  # Adjust the spacing between subplots
    for ax in axes.flat:
        # Make the spines thinner
        for spine in ax.spines.values():
            spine.set_linewidth(curr_linewidth)  # Adjust the linewidth as needed

    # Get data per goal projection
    for i_goal in plot_lines:
        x_vals = np.array(results_dict['{}_{}'.format(plot_dims[0],str(i_goal))])
        y_vals = np.array(results_dict['{}_{}'.format(plot_dims[1],str(i_goal))])

        # Plot data per time period
        for window_group_num, window_group_item in enumerate(window_groups_plot.items()):
            window_group_name = window_group_item[0]
            window_group_plot_pos =window_group_item[1]
            window_group_num = window_group_num + 2

            axes[window_group_plot_pos[0],window_group_plot_pos[1]].scatter(
                x_vals[window_group_num], y_vals[window_group_num], color=line_colours[i_goal], s=4
            )

    # Add legend
    for i_goal in plot_lines:
        axes[0,0].plot([], [],'-', marker='o', color=line_colours[i_goal], alpha=1, label=str(i_goal), markersize=4)
    axes[0,0].legend(loc='center right', bbox_to_anchor=(1.3, 0.5))

    # Add titles to subplots
    x_grid_titles = ['Early Choice', 'Late Choice', 'Go']
    for ax, col in zip(axes[0], x_grid_titles):
        ax.set_title(col, size=BIGGER_SIZE)
    y_grid_titles = ['C1', 'C2 (goal)', 'C2 (away)', 'C3', 'C4 (goal)']
    for ax, row in zip(axes[:,0], y_grid_titles):
        ax.set_ylabel(row, rotation=90, size=BIGGER_SIZE)

    # Add the same x and y axis limits to all subplots, scale them dynamically to min and max across all subplots
    x_min = np.min([ax.get_xlim()[0] for ax in axes.flatten()])
    x_max = np.max([ax.get_xlim()[1] for ax in axes.flatten()])
    y_min = np.min([ax.get_ylim()[0] for ax in axes.flatten()])
    y_max = np.max([ax.get_ylim()[1] for ax in axes.flatten()])
    for ax in axes.flatten():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Hide the number on axis except for on lowest row and on first / left column
    for ax in axes.flatten():
        if ax in axes[4]:
            ax.set_xlabel('PCA dim 1')
        else:
            ax.tick_params(labelbottom=False)
        if ax in axes[:,0]:
            pass
        else:
            ax.tick_params(labelleft=False)

    # Save figure
    # Make font editable in svg
    plt.rcParams['svg.fonttype'] = 'none'
    # Set font to arial
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'arial'
    plt.savefig(resultsLoc+'print/print_{0}.svg'.format(plot_name), format='svg',dpi=300, bbox_inches="tight")

    plt.close()


def EV_for_set_of_windows(data_dict, grouped_projection_windows, pca, pivot_column = 'FinalPos'):
    '''
    Project each window in grouped_projection_windows onto the PCA dimensions.
    '''
    ## Create dict to hold results
    num_pca_dims = pca.components_.shape[0]
    dim_names = ['dim'+str(i) for i in range(1,num_pca_dims+1)]
    results_listI = []
    results_listII = []

    ## Project onto dims
    for window_name, window_params in grouped_projection_windows.items():
        # Get data
        test_goal_mean_df = data_dict[window_params[0]]
        temp_pop_df = test_goal_mean_df[['Unit',pivot_column,window_name]]
        temp_pop_df = temp_pop_df.pivot(index='Unit', columns=pivot_column, values=window_name)
        temp_pop_array = temp_pop_df.values.T

        # Project data
        try:
            proj_data = pca.transform(temp_pop_array)

            # Calculate explained variance
            inv_proj_data = pca.inverse_transform(proj_data)
            varexp = explained_variance_score(temp_pop_array,inv_proj_data, multioutput='variance_weighted')

            # Add results to list
            results_listI.append(varexp)

            # Alternative EV procedure; currently deactivated
            #results_listII.append(neuron_corr_ev(temp_pop_array,proj_data))
            results_listII.append(0)
        except:
            results_listI.append(0)
            results_listII.append(0)
            print('Error in window: {0}'.format(window_name))

    return results_listI, results_listII


def EV_from_goal_dimensions(curr_area_sweep, grouping_var = 'FinalPos', goal_dim_fun = None, windows_for_dims_object = windows_for_dims_dicts):
    # Create data subselection
    analysisdf, unit_list, data_area = create_data_subselection(curr_area_sweep, analysisdf_import)
    analysisdf.reset_index(inplace=True, drop=True)

    # Split analysisdf into two halves, using sklearn and stratified by unit, goal
    #analysisdf['statification_c4'] = analysisdf['Choice4step'].map({1:1})
    #analysisdf['statification_var'] = analysisdf['Unit'].astype(str)+'_'+analysisdf[grouping_var].astype(str)+'_'+analysisdf['statification_c4'].astype(str)
    analysisdf['statification_var'] = analysisdf['Unit'].astype(str)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=23)
    analysisdf_halves = []
    for train_index, test_index in splitter.split(analysisdf, analysisdf['statification_var']):
        analysisdf_halves.append(analysisdf.iloc[train_index])
        analysisdf_halves.append(analysisdf.iloc[test_index])

    # Create PCA and data on each half
    component_dict = {}
    for i_half, i_analysisdf in enumerate(analysisdf_halves):
        pca, _ = goal_dim_fun(i_analysisdf,windows_for_dims_object, unit_list = unit_list, pca_components=2)
        data_dict = create_projection_data(i_analysisdf, grouped_projection_windows, windows_for_projection, unit_list = unit_list, projection_groups = ['Unit',grouping_var])
        # Add to dict
        component_dict['pca_'+str(i_half)]=pca
        component_dict['data_'+str(i_half)]=data_dict

    # Project data onto PCA components, considering test and train data
    results_dict = {}
    return_dict = {}
    for i_key in ['0_1','1_0']:
        # Get data
        i_pca, i_data = i_key.split('_')
        i_pca = component_dict['pca_'+i_pca]
        i_data = component_dict['data_'+i_data]
        # Project
        results_dict[i_key+'_I'],results_dict[i_key+'_II']  = EV_for_set_of_windows(i_data, grouped_projection_windows, i_pca, pivot_column = grouping_var)

    # Mean of both lists in results_dict
    return_dict['mean_revProj'] = np.mean([results_dict['0_1_I'],results_dict['1_0_I']],axis=0)
    return_dict['mean_Corr'] = np.mean([results_dict['0_1_II'],results_dict['1_0_II']],axis=0)

    return return_dict, component_dict, analysisdf_halves, unit_list

def pval_EV_from_goal_dimensions(curr_area_sweep, grouping_var = 'FinalPos', goal_dim_fun = None, windows_for_dims_object = windows_for_dims_dicts, signficance_iteration_samples = 1000, component_dict = None, main_analysis_dict = None, analysisdf_halves = None, unit_list = None):
    '''
    Get p values for explained variance for each window for the goal dimension.
    '''
    # Create null distribution
    null_results_dict = {
        'mean_revProj':[],
        'mean_Corr':[]}

    for i_null in range(signficance_iteration_samples):
        for i_half, i_analysisdf in enumerate(analysisdf_halves):
            data_dict = create_projection_data(i_analysisdf, grouped_projection_windows, windows_for_projection, unit_list =unit_list, projection_groups = ['Unit',grouping_var], shuffle_group=True)
            component_dict['data_'+str(i_half)]=data_dict

        # Project data onto PCA components, considering test and train data
        results_dict = {}
        for i_key in ['0_1','1_0']:
            # Get data
            i_pca, i_data = i_key.split('_')
            i_pca = component_dict['pca_'+i_pca]
            i_data = component_dict['data_'+i_data]
            # Project
            results_dict[i_key+'_I'],results_dict[i_key+'_II']  = EV_for_set_of_windows(i_data, grouped_projection_windows, i_pca, pivot_column = grouping_var)

        # Mean of both lists in results_dict
        null_results_dict['mean_revProj'].append(np.mean([results_dict['0_1_I'],results_dict['1_0_I']],axis=0))
        null_results_dict['mean_Corr'].append(np.mean([results_dict['0_1_II'],results_dict['1_0_II']],axis=0))
        print(i_null)

    # Iterate over all windows to extract p values
    p_values_list = []
    for i_window in range(len(grouped_projection_windows)):
        # Get data
        main_projection_value = main_analysis_dict['mean_revProj'][i_window]
        null_projection_values = [i[i_window] for i in null_results_dict['mean_revProj']]
        # Get rank of main value in null distribution
        sorted_null_projection_values = sorted(null_projection_values)
        rank_value = np.searchsorted(sorted_null_projection_values,main_projection_value)
        # Calculate p value
        rank_value = np.abs(rank_value-(signficance_iteration_samples/2))-1
        p_val_per_rank = 1/(signficance_iteration_samples/2)
        p_value = 1-(rank_value*p_val_per_rank)
        # Add to list
        p_values_list.append(p_value)

    fdr_p_values = fdrcorrection(p_values_list, alpha = 0.05)
    return fdr_p_values

from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes 
def align_pca_results(original_dict, to_align_dict, limit_dimensions = 2, limit_transform=True):
    # Determine the number of dimensions, conditions, and time windows
    if type(limit_dimensions) == int:
        n_dims = limit_dimensions
    else:
        n_dims = max([int(key.split('_')[0][3:]) for key in original_dict.keys()])
    conditions = sorted(set([int(key.split('_')[1]) for key in original_dict.keys()]))
    n_time_windows = len(original_dict['dim1_' + str(conditions[0])])

    aligned_dict = {}

    for t in range(n_time_windows):
        # Reconstruct original PCA matrix for this time window
        original_matrix = np.array([[original_dict[f'dim{d}_{c}'][t] for d in range(1, n_dims+1)] 
                                    for c in conditions])

        # Reconstruct to-be-aligned PCA matrix for this time window
        to_align_matrix = np.array([[to_align_dict[f'dim{d}_{c}'][t] for d in range(1, n_dims+1)] 
                                    for c in conditions])

        # Perform Procrustes alignment
        if limit_transform:
            R, _ = orthogonal_procrustes(to_align_matrix, original_matrix)
            aligned_matrix = to_align_matrix @ R
        else:
            _, aligned_matrix , _ = procrustes(original_matrix, to_align_matrix)
        
        # Store aligned results back in dictionary format
        for i, c in enumerate(conditions):
            for d in range(1, n_dims+1):
                key = f'dim{d}_{c}'
                if key not in aligned_dict:
                    aligned_dict[key] = []
                aligned_dict[key].append(aligned_matrix[i, d-1])

    return aligned_dict

def confidence_ellipse(x, y, ax, n_dev=2.0, type_of_dev='std', facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Based on: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    Adapted to allow plotting standard errors and F-distribution based confidence intervals.
    
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_dev : float
        For 'std' and 'std_error': number of standard deviations
        For 'conf': confidence level (e.g., 0.95 for 95% confidence)
    type_of_dev : str
        Type of deviation to plot:
        'std' - standard deviation
        'std_error' - standard error
        'conf' - F-distribution based confidence interval
    facecolor : str
        Color to fill the ellipse
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
        
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    n_pop = x.size
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                     facecolor=facecolor, fill=True, linewidth=0, **kwargs)
    
    # Calculate scaling factors based on the type of deviation
    if type_of_dev == 'std':
        scale_x = np.sqrt(cov[0, 0]) * n_dev
        scale_y = np.sqrt(cov[1, 1]) * n_dev
    elif type_of_dev == 'std_error':
        scale_x = np.sqrt(cov[0, 0])/np.sqrt(n_pop) * n_dev
        scale_y = np.sqrt(cov[1, 1])/np.sqrt(n_pop) * n_dev
    elif type_of_dev == 'conf':
        # F-distribution based confidence interval - matching MATLAB implementation
        p = 2  # dimensionality (2D data)
        # Calculate k exactly as in MATLAB
        k = stats.f.ppf(n_dev, p, n_pop-p) * p*(n_pop-1)/(n_pop-p)
        # Take square root of k times the variance
        scale_x = np.sqrt(k * cov[0, 0])
        scale_y = np.sqrt(k * cov[1, 1])
    else:
        raise ValueError("type_of_dev must be one of: 'std', 'std_error', 'conf'")
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def create_grid_bootstrapped_plot(boot_results_dict, plot_lines, plot_name, resultsLoc):
    # Subselect data
    plot_dims = ['dim1','dim2']

    # Plot assignment
    window_groups_plot = {
        'EarlyGoal':[0,0],
        'LateGoal':[0,1],
        'EarlyC1':[1,0],
        'LateC1':[1,1],
        'GoC1':[1,2],
        'EarlyC2_2step':[2,0],
        'LateC2_2step':[2,1],
        'GoC2_2step':[2,2],
        'EarlyC2_4step':[3,0],
        'LateC2_4step':[3,1],
        'GoC2_4step':[3,2],
        'EarlyC3_4step':[4,0],
        'LateC3_4step':[4,1],
        'GoC3_4step':[4,2],
        'EarlyC4_4step':[5,0],
        'LateC4_4step':[5,1],
        'GoC4_4step':[5,2]
        }
    windows_for_projection = list(window_groups_plot.keys())

    ## Get 4 RGB codes from viridis colormap
    colours = plt.cm.viridis(np.linspace(0,1,len(plot_lines)))
    line_colours = dict(zip(plot_lines,colours))

    # Setup figure
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 14))
    st = fig.suptitle(plot_name)

    # Get data per goal projection
    for i_goal in plot_lines:
        x_vals = np.array(boot_results_dict['{}_{}'.format(plot_dims[0],str(i_goal))])
        y_vals = np.array(boot_results_dict['{}_{}'.format(plot_dims[1],str(i_goal))])

        # Plot data per time period
        for window_group_num, window_group_item in enumerate(window_groups_plot.items()):
            window_group_name = window_group_item[0]
            window_group_plot_pos =window_group_item[1]

            # Select plot data
            x_vals_plot = x_vals[:,window_group_num]
            y_vals_plot = y_vals[:,window_group_num]

            axes[window_group_plot_pos[0],window_group_plot_pos[1]].scatter(
                x_vals_plot, y_vals_plot, color=line_colours[i_goal], alpha=0.0
            )

            # Add confidence ellipse
            confidence_ellipse(x_vals_plot, y_vals_plot, axes[window_group_plot_pos[0],window_group_plot_pos[1]], edgecolor=line_colours[i_goal], facecolor=line_colours[i_goal], alpha=0.5)


    # Add legend
    for i_goal in plot_lines:
        axes[0,0].plot([], [],'-', marker='o', color=line_colours[i_goal], alpha=0.5, label=str(i_goal))
    axes[0,0].legend(loc='center right', bbox_to_anchor=(1.3, 0.5))

    # Set subplot information
    for window_group_num, window_group_item in enumerate(window_groups_plot.items()):
        window_group_name = window_group_item[0]
        window_group_plot_pos =window_group_item[1]
        axes[window_group_plot_pos[0],window_group_plot_pos[1]].set_xlabel(plot_dims[0])
        axes[window_group_plot_pos[0],window_group_plot_pos[1]].set_ylabel(plot_dims[1])

        # Add title to axes
        axes[window_group_plot_pos[0],window_group_plot_pos[1]].set_title(window_group_name)


    # Add the same x and y axis limits to all subplots, scale them dynamically to min and max across all subplots
    x_min = np.min([ax.get_xlim()[0] for ax in axes.flatten()])
    x_max = np.max([ax.get_xlim()[1] for ax in axes.flatten()])
    y_min = np.min([ax.get_ylim()[0] for ax in axes.flatten()])
    y_max = np.max([ax.get_ylim()[1] for ax in axes.flatten()])
    for ax in axes.flatten():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Save figure
    plt.tight_layout()
    plt.savefig(resultsLoc+'boot_{0}.png'.format(plot_name),dpi = 300)
    plt.close()

############ Execution calls ############

def simple_projection_routine(curr_area_sweep = areas_sweep[0], save_figure = False, return_df = False):
    '''
    Basic routine to create projection plots.
    
    # Apply loop
    for i_area_sweep in areas_sweep:
        _ = simple_projection_routine(curr_area_sweep = i_area_sweep, save_figure = True)
    '''
    analysisdf, unit_list, _ = create_data_subselection(curr_area_sweep, analysisdf_import)
    pca, data_df = create_GoalDimensions_C2C4(analysisdf,windows_for_dims_dicts, unit_list = unit_list, pca_components=pca_components, return_df = return_df)
    data_dict = create_projection_data(analysisdf, grouped_projection_windows, windows_for_projection, unit_list = unit_list, projection_groups = ['Unit','FinalPos'])
    results_dict = project_set_of_windows(data_dict, grouped_projection_windows, pca, goal_numbers)
    if save_figure:
        plot_name = 'SimpleGrid_{0}'.format(curr_area_sweep[1])
        create_grid_plot_overview(results_dict, goal_numbers, plot_name, resultsLoc)
        create_grid_plot_print(results_dict, goal_numbers, plot_name, resultsLoc)

    if return_df:
        return results_dict, pca, data_dict, data_df
    else:
        return results_dict, pca, data_dict

def boostrapped_procrustes_projection_routine(curr_area_sweep = areas_sweep[0], master_projection = False):
    #curr_area_sweep = areas_sweep[1]
    # Create baseline PCA
    analysisdf, unit_list, _ = create_data_subselection(curr_area_sweep, analysisdf_import)
    pca, data_df = create_GoalDimensions_C2C4(analysisdf,windows_for_dims_dicts, unit_list = unit_list, pca_components=pca_components, return_df = True)
    data_dict = create_projection_data(analysisdf, grouped_projection_windows, windows_for_projection, unit_list = unit_list, projection_groups = ['Unit','FinalPos'])
    baseline_results_dict = project_set_of_windows(data_dict, grouped_projection_windows, pca, goal_numbers)
    # Get reference projection for master projection
    if master_projection:
        master_data_dict = {'master':data_df.reset_index()}
        reference_master_results_dict = project_set_of_windows(master_data_dict, {'dim_window':('master',['dim_window'])}, pca, goal_numbers)

    # Create dict to hold bootstrap results
    num_pca_dims = pca.components_.shape[0]
    dim_names = ['dim'+str(i) for i in range(1,num_pca_dims+1)]
    boot_results_dict = {}
    for i_goal in goal_numbers:
        for i_key in dim_names:
            boot_results_dict[i_key+'_'+str(i_goal)]=[]
    master_boot_results_dict = {}
    for i_goal in goal_numbers:
        for i_key in dim_names:
            master_boot_results_dict[i_key+'_'+str(i_goal)]=[]

    # Created bootstrapped & aligned projections
    analysisdf.reset_index(inplace=True, drop=True)
    stratification_var = analysisdf['Unit'].astype(str)
    for i_split in range(1000):
        sampled_index = resample(analysisdf.index, replace=True, n_samples=len(analysisdf), stratify=stratification_var)
        print(f"Fold {i_split}:")
        # Subselect data and project
        analysisdf_boot = analysisdf.iloc[sampled_index].reset_index(drop=True)
        pca_boot, pca_boot_trainingdata = create_GoalDimensions_C2C4(analysisdf_boot,windows_for_dims_dicts, unit_list = unit_list, pca_components=pca_components, return_df = True)
        data_dict = create_projection_data(analysisdf_boot, grouped_projection_windows, windows_for_projection, unit_list = unit_list, projection_groups = ['Unit','FinalPos'])
        results_dict = project_set_of_windows(data_dict, grouped_projection_windows, pca_boot, goal_numbers)
        aligned_results_dict = align_pca_results(baseline_results_dict, results_dict, limit_transform=True)
        #aligned_results_dict = results_dict
        ## Add each list from results_dict to boot_results_dict
        for key, value in aligned_results_dict.items():
            boot_results_dict[key].append(value)

        if master_projection:
            master_data_dict = {'master':pca_boot_trainingdata.reset_index()}
            master_results_dict = project_set_of_windows(master_data_dict, {'dim_window':('master',['dim_window'])}, pca_boot, goal_numbers)
            master_aligned_results_dict = align_pca_results(reference_master_results_dict, master_results_dict, limit_transform=True)
            for key, value in master_aligned_results_dict.items():
                master_boot_results_dict[key].append(value[0])
    create_grid_bootstrapped_plot(boot_results_dict, goal_numbers, '{0}_{1}'.format(curr_area_sweep[1], 'full'), resultsLoc)
    if master_projection:
        simple_scatter_ellipse_plot(master_boot_results_dict, goal_numbers, 'master_{0}'.format(curr_area_sweep[1]), resultsLoc)
    return boot_results_dict

def EV_GoalDimensions_C2C4_plot(curr_area_sweep = areas_sweep[0], signficance_iteration_samples = 1000):
    # Create EV projections and their p values
    results_dict, component_dict, analysisdf_halves, unit_list = EV_from_goal_dimensions(
        curr_area_sweep = curr_area_sweep, grouping_var = 'FinalPos', goal_dim_fun = create_GoalDimensions_C2C4, windows_for_dims_object = windows_for_dims_dicts)
    p_values = pval_EV_from_goal_dimensions(
        curr_area_sweep = curr_area_sweep, grouping_var = 'FinalPos', goal_dim_fun = None, windows_for_dims_object = windows_for_dims_dicts, 
        signficance_iteration_samples = signficance_iteration_samples, component_dict = component_dict,
        main_analysis_dict = results_dict, analysisdf_halves = analysisdf_halves, unit_list = unit_list)
    # Get values for plot
    x_vals = np.arange(15)
    y_vals_key ='mean_revProj'
    y_vals = results_dict[y_vals_key]

    # Save p values and ev to disk
    evdf = {'window':np.array(list(grouped_projection_windows.keys())) ,'ev':y_vals, 'p_values':p_values[1]}
    evdf = pd.DataFrame(evdf)
    evdf.to_csv(resultsLoc+'EV_{0}.csv'.format(curr_area_sweep[1]))

    y_vals = y_vals[2:]
    y_vals_sig = np.ma.masked_where(~p_values[0][2:], y_vals)

    # Set the figure size to 3x3 inches
    fig, ax = plt.subplots(figsize=(2, 2))

    # Plot underlying data with grey, and significant data on top in color
    ax.plot(x_vals, y_vals, 'o-', color='lightgrey', markersize=4, label='Not significant')
    ax.plot(x_vals, y_vals_sig, 'o-', color='#8B2235', markersize=4, label='Significant')

    # Set y axis limits
    ax.set_ylim(0, 0.75)

    # Other plot detail
    ax.set_xticks(x_vals)
    ax.set_xticklabels(grouped_projection_windows_cleanedNames.values(), rotation='vertical')
    ax.set_ylabel('Explained variance')
    ax.set_xlabel('Time period')
    ax.legend()
    ax.set_title(curr_area_sweep[1])
    curr_linewidth = 0.5
    # Make the spines thinner
    for spine in ax.spines.values():
        spine.set_linewidth(curr_linewidth)  # Adjust the linewidth as needed

    # Save the figure
    plt.tight_layout()
    #Make font editable in svg
    plt.rcParams['svg.fonttype'] = 'none'
    # Set font to arial
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'arial'
    # Save figure
    plt.savefig(resultsLoc+'print/EV_sig_{0}.svg'.format(curr_area_sweep[1]), format='svg',dpi=300, bbox_inches="tight")
    plt.savefig(resultsLoc + 'EV_sig_{0}.png'.format(curr_area_sweep[1]), dpi=300)
    plt.close()

    return y_vals, y_vals_sig


def manuscripts_plot(results_dict, boot_results_dict, ev_data, plot_lines, plot_name, resultsLoc):
    # Subselect data
    plot_dims = ['dim1','dim2']

    # Plot assignment
    projection_plot_assignment = {
        'EarlyC1':[0,0],
        'LateC1': [0,1],
        'GoC1':[0,2],
        'EarlyC2_2step':[1,0],
        'LateC2_2step':[1,1],
        'GoC2_2step':[1,2],
        'EarlyC2_4step':None,
        'LateC2_4step':None,
        'GoC2_4step':None,
        'EarlyC3_4step':[2,0],
        'LateC3_4step':[2,1],
        'GoC3_4step':[2,2],
        'EarlyC4_4step':[3,0],
        'LateC4_4step':[3,1],
        'GoC4_4step':[3,2]
        }
    ev_plot_assignment = {
        0: ['EarlyC1','LateC1','GoC1'],
        1: ['EarlyC2_2step','LateC2_2step','GoC2_2step'],
        2: ['EarlyC3_4step','LateC3_4step','GoC3_4step'],
        3: ['EarlyC4_4step','LateC4_4step','GoC4_4step']
        }
    windows_for_projection = list(projection_plot_assignment.keys())

    # Get 4 RGB codes from viridis colormap
    colours = plt.cm.viridis(np.linspace(0,1,len(plot_lines)))
    line_colours = dict(zip(plot_lines,colours))

    # Create font information 
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 6,6,7
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Setup figure
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(3.0, 3.0))
    st = fig.suptitle(plot_name)

    # Adjust general plot parameters
    curr_linewidth = 0.5
    plt.subplots_adjust(hspace=0.3)  # Adjust the spacing between subplots
    for ax in axes.flat:
        # Make the spines thinner
        for spine in ax.spines.values():
            spine.set_linewidth(curr_linewidth)  # Adjust the linewidth as needed

    '''
    # Add projection plots
    for i_goal in plot_lines:
        x_vals = np.array(results_dict['{}_{}'.format(plot_dims[0],str(i_goal))])
        y_vals = np.array(results_dict['{}_{}'.format(plot_dims[1],str(i_goal))])

        # Plot data per time period
        for window_group_num, window_group_item in enumerate(projection_plot_assignment.items()):
            if window_group_item[1] is None:
                pass
            else:
                window_group_name = window_group_item[0]
                window_group_plot_pos =window_group_item[1]
                window_group_num = window_group_num + 2

                axes[window_group_plot_pos[0],window_group_plot_pos[1]].scatter(
                    x_vals[window_group_num], y_vals[window_group_num], color=line_colours[i_goal], s=4
                )
    '''

    # Get data per goal projection
    for i_goal in plot_lines:
        x_vals = np.array(boot_results_dict['{}_{}'.format(plot_dims[0],str(i_goal))])
        y_vals = np.array(boot_results_dict['{}_{}'.format(plot_dims[1],str(i_goal))])

        # Plot data per time period
        for window_group_num, window_group_item in enumerate(projection_plot_assignment.items()):
            if window_group_item[1] is None:
                pass
            else:
                window_group_name = window_group_item[0]
                window_group_plot_pos =window_group_item[1]

                # Select plot data
                window_group_num = window_group_num + 2
                x_vals_plot = x_vals[:,window_group_num]
                y_vals_plot = y_vals[:,window_group_num]

                axes[window_group_plot_pos[0],window_group_plot_pos[1]].scatter(
                    x_vals_plot, y_vals_plot, color=line_colours[i_goal], alpha=0.0
                )

                # Add confidence ellipse
                confidence_ellipse(x_vals_plot, y_vals_plot, axes[window_group_plot_pos[0],window_group_plot_pos[1]], n_dev=0.95, type_of_dev = 'conf', facecolor=line_colours[i_goal], alpha=1.0)


    # Add ev plots
    for i_row, i_ev_numbers in ev_plot_assignment.items():
        # Get indices corresponding to the ev numbers
        ev_windows_for_projection = windows_for_projection
        i_ev_indices = [ev_windows_for_projection.index(i) for i in i_ev_numbers]

        # Get data
        x_vals = np.arange(len(i_ev_indices))
        y_vals = np.array(ev_data[0])[i_ev_indices]
        y_vals_sig = ev_data[1][i_ev_indices]
        #print(ev_data[1])
        #print(np.array(ev_data[1]))

        # Plot data
        axes[i_row,3].plot(x_vals, y_vals, 'o-', color='lightgrey', markersize=2, linewidth=0.75, label='Not significant')
        axes[i_row,3].plot(x_vals, y_vals_sig, 'o-', color='#8B2235', markersize=2, linewidth=0.75, label='Significant')
        axes[i_row,3].set_ylim(0, 0.75)

        # Other plot detail
        axes[i_row,3].set_xticks(x_vals)
        if i_row == 2:
            axes[i_row,3].set_ylabel('Explained variance')
        if i_row == 3:
            axes[i_row,3].set_xticklabels(['Early', 'Late', 'Go'], rotation='vertical')
            axes[i_row,3].set_xlabel('Time period')
        else:
            axes[i_row,3].set_xticklabels([])

        # Add legend above first plot in column
        if i_row == 0:
            axes[i_row,3].legend(loc='center right', bbox_to_anchor=(0, 1.5))

    # Add titles to subplots
    x_grid_titles = ['Early Choice', 'Late Choice', 'Go', ' ']
    for ax, col in zip(axes[0], x_grid_titles):
        ax.set_title(col, size=BIGGER_SIZE)
    y_grid_titles = ['C1', 'C2', 'C3', 'C4']
    for ax, row in zip(axes[:,0], y_grid_titles):
        ax.set_ylabel(row, rotation=90, size=BIGGER_SIZE)

    # Add the same x and y axis limits to all subplots in the first 3 columns, scale them dynamically to min and max across all subplots in the first 3 columns
    global_minmax = True
    x_min = np.min([ax.get_xlim()[0] for ax in axes[:,:3].flatten()])
    x_max = np.max([ax.get_xlim()[1] for ax in axes[:,:3].flatten()])
    y_min = np.min([ax.get_ylim()[0] for ax in axes[:,:3].flatten()])
    y_max = np.max([ax.get_ylim()[1] for ax in axes[:,:3].flatten()])
    if global_minmax:
        g_min = np.min([x_min, y_min])
        g_max = np.max([x_max, y_max])
        for ax in axes[:,:3].flatten():
            ax.set_xlim(g_min, g_max)
            ax.set_ylim(g_min, g_max)
    else:
        for ax in axes[:,:3].flatten():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    # Hide numbers on x and y axis for all plots in first 3 columns, except for on lowest row and on first / left column
    for ax in axes[:,:3].flatten():
        if ax in axes[3]:
            ax.set_xlabel('PCA Dim. 1')
        else:
            ax.tick_params(labelbottom=False)
        if ax in axes[:,0]:
            pass
        else:
            ax.tick_params(labelleft=False)

    # Add legend
    for i_goal in plot_lines:
        axes[0,0].plot([], [],'-', marker='o', color=line_colours[i_goal], alpha=1, label=str(i_goal), markersize=4)
    axes[0,0].legend(loc='center right', bbox_to_anchor=(1.3, 0.5))

    # Save png
    #plt.tight_layout()
    plt.savefig(resultsLoc+'{0}.png'.format(plot_name),dpi = 300)

    # Save figure
    # Make font editable in svg
    plt.rcParams['svg.fonttype'] = 'none'
    # Set font to arial
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'arial'
    plt.savefig(resultsLoc+'print/print_{0}.svg'.format(plot_name), format='svg',dpi=300)
    plt.close()

def manuscripts_plot_4step(results_dict, boot_results_dict, ev_data, plot_lines, plot_name, resultsLoc):
    # Subselect data
    plot_dims = ['dim1','dim2']

    # Plot assignment
    projection_plot_assignment = {
        'EarlyC1':None,
        'LateC1': None,
        'GoC1':None,
        'EarlyC2_2step':None,
        'LateC2_2step':None,
        'GoC2_2step':None,
        'EarlyC2_4step':[0,0],
        'LateC2_4step':[0,1],
        'GoC2_4step':[0,2],
        'EarlyC3_4step':None,
        'LateC3_4step':None,
        'GoC3_4step':None,
        'EarlyC4_4step':None,
        'LateC4_4step':None,
        'GoC4_4step':None
        }
    ev_plot_assignment = {
        0: ['EarlyC2_4step','LateC2_4step','GoC2_4step']
        }
    windows_for_projection = list(projection_plot_assignment.keys())

    # Get 4 RGB codes from viridis colormap
    colours = plt.cm.viridis(np.linspace(0,1,len(plot_lines)))
    line_colours = dict(zip(plot_lines,colours))

    # Create font information 
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 6,6,7
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Setup figure
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(5.5, 2.0))
    axes = axes.reshape(1, 4)  # Reshape to 2D array with shape (1, 4)
    st = fig.suptitle(plot_name)

    # Adjust general plot parameters
    curr_linewidth = 0.5
    #plt.subplots_adjust(hspace=0.3)  # Adjust the spacing between subplots
    for ax in axes.flat:
        # Make the spines thinner
        for spine in ax.spines.values():
            spine.set_linewidth(curr_linewidth)  # Adjust the linewidth as needed

    # Get data per goal projection
    for i_goal in plot_lines:
        x_vals = np.array(boot_results_dict['{}_{}'.format(plot_dims[0],str(i_goal))])
        y_vals = np.array(boot_results_dict['{}_{}'.format(plot_dims[1],str(i_goal))])

        # Plot data per time period
        for window_group_num, window_group_item in enumerate(projection_plot_assignment.items()):
            if window_group_item[1] is None:
                pass
            else:
                window_group_name = window_group_item[0]
                window_group_plot_pos =window_group_item[1]

                # Select plot data
                window_group_num = window_group_num + 2
                x_vals_plot = x_vals[:,window_group_num]
                y_vals_plot = y_vals[:,window_group_num]

                axes[window_group_plot_pos[0],window_group_plot_pos[1]].scatter(
                    x_vals_plot, y_vals_plot, color=line_colours[i_goal], alpha=0.0
                )

                # Add confidence ellipse
                confidence_ellipse(x_vals_plot, y_vals_plot, axes[window_group_plot_pos[0],window_group_plot_pos[1]], n_dev=0.95, type_of_dev = 'conf', facecolor=line_colours[i_goal], alpha=1.0)


    # Add ev plots
    for i_row, i_ev_numbers in ev_plot_assignment.items():
        # Get indices corresponding to the ev numbers
        ev_windows_for_projection = windows_for_projection
        i_ev_indices = [ev_windows_for_projection.index(i) for i in i_ev_numbers]

        # Get data
        x_vals = np.arange(len(i_ev_indices))
        y_vals = np.array(ev_data[0])[i_ev_indices]
        y_vals_sig = ev_data[1][i_ev_indices]

        # Plot data
        axes[i_row,3].plot(x_vals, y_vals, 'o-', color='lightgrey', markersize=2, linewidth=0.75, label='Not significant')
        axes[i_row,3].plot(x_vals, y_vals_sig, 'o-', color='#8B2235', markersize=2, linewidth=0.75, label='Significant')
        axes[i_row,3].set_ylim(0, 0.75)

        # Other plot detail
        axes[i_row,3].set_xticks(x_vals)
        axes[i_row,3].set_ylabel('Explained variance')
        axes[i_row,3].set_xticklabels(['Early', 'Late', 'Go'], rotation='vertical')
        axes[i_row,3].set_xlabel('Time period')

        # Add legend above first plot in column
        if i_row == 0:
            axes[i_row,3].legend(loc='center right', bbox_to_anchor=(0, 1.5))

    # Add titles to subplots
    x_grid_titles = ['Early Choice', 'Late Choice', 'Go', ' ']
    for ax, col in zip(axes[0], x_grid_titles):
        ax.set_title(col, size=BIGGER_SIZE)
    #y_grid_titles = ['C1', 'C2', 'C3', 'C4']
    #for ax, row in zip(axes[:,0], y_grid_titles):
    #    ax.set_ylabel(row, rotation=90, size=BIGGER_SIZE)

    # Add the same x and y axis limits to all subplots in the first 3 columns, scale them dynamically to min and max across all subplots in the first 3 columns
    global_minmax = True
    x_min = np.min([ax.get_xlim()[0] for ax in axes[:,:3].flatten()])
    x_max = np.max([ax.get_xlim()[1] for ax in axes[:,:3].flatten()])
    y_min = np.min([ax.get_ylim()[0] for ax in axes[:,:3].flatten()])
    y_max = np.max([ax.get_ylim()[1] for ax in axes[:,:3].flatten()])
    if global_minmax:
        g_min = np.min([x_min, y_min])
        g_max = np.max([x_max, y_max])
        for ax in axes[:,:3].flatten():
            ax.set_xlim(g_min, g_max)
            ax.set_ylim(g_min, g_max)
    else:
        for ax in axes[:,:3].flatten():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    # Hide y-axis labels for columns 2 and 3 
    axes[0,1].tick_params(labelleft=False)
    axes[0,2].tick_params(labelleft=False)

    # Set x labels for all plots
    for i in range(3):
        axes[0,i].set_xlabel('PCA Dim. 1')

    # Set y label only for first column
    axes[0,0].set_ylabel('PCA Dim. 2')

    # Add legend
    for i_goal in plot_lines:
        axes[0,0].plot([], [],'-', marker='o', color=line_colours[i_goal], alpha=1, label=str(i_goal), markersize=4)
    axes[0,0].legend(loc='center right', bbox_to_anchor=(1.3, 0.5))

    # Save png
    plt.tight_layout(pad = 0.001)
    plt.savefig(resultsLoc+'print4/{0}.png'.format(plot_name),dpi = 300)

    # Save figure
    # Make font editable in svg
    plt.rcParams['svg.fonttype'] = 'none'
    # Set font to arial
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'arial'
    plt.savefig(resultsLoc+'print4/print_{0}.svg'.format(plot_name), format='svg',dpi=300, bbox_inches="tight")
    plt.close()

def simple_scatter_ellipse_plot(data_dict, plot_lines, plot_name, resultsLoc):
    """
    Create a scatter plot with confidence ellipses for different groups.
    """
    # Create font information 
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 6, 6, 7
    plt.rc('font', size=SMALL_SIZE)          
    plt.rc('axes', titlesize=SMALL_SIZE)     
    plt.rc('axes', labelsize=MEDIUM_SIZE)    
    plt.rc('xtick', labelsize=SMALL_SIZE)    
    plt.rc('ytick', labelsize=SMALL_SIZE)    
    plt.rc('legend', fontsize=SMALL_SIZE)    
    plt.rc('figure', titlesize=BIGGER_SIZE)  

    # Setup figure
    fig, ax = plt.subplots(figsize=(0.75, 0.75))
    
    # Get colors from viridis colormap
    colours = plt.cm.viridis(np.linspace(0, 1, len(plot_lines)))
    line_colours = dict(zip(plot_lines, colours))
    
    # Plot data for each group
    for i_goal in plot_lines:
        x_data = np.array(data_dict['dim1_'+str(i_goal)])
        y_data = np.array(data_dict['dim2_'+str(i_goal)])
        
        # Add scatter plot (with alpha=0 to hide points but keep for reference)
        ax.scatter(x_data, y_data, color=line_colours[i_goal], alpha=0.0)
        
        # Add confidence ellipse
        confidence_ellipse(x_data, y_data, ax, edgecolor=line_colours[i_goal],n_dev=0.95, type_of_dev = 'conf', facecolor=line_colours[i_goal], alpha=1.0)
    
    # Set same limits for both axes
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    global_min = min(x_min, y_min)
    global_max = max(x_max, y_max)
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(global_min, global_max)

    # Set the same ticks for both axes (maximum 5 ticks)
    tick_locator = plt.MaxNLocator(5)
    ticks = tick_locator.tick_values(global_min, global_max)
    # Set positions and labels
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    # Add labels
    ax.set_xlabel('PCA Dim. 1')
    ax.set_ylabel('PCA Dim. 2')
    
    # Save png
    plt.savefig(resultsLoc + '{0}.png'.format(plot_name), dpi=300)

    # Save figure
    # Make font editable in svg
    plt.rcParams['svg.fonttype'] = 'none'
    # Set font to arial
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'arial'
    plt.savefig(resultsLoc + 'print/print_{0}.svg'.format(plot_name), format='svg', dpi=300)
    plt.close()


if __name__ == "__main__":
    print('dimsSpikeGoalI_2ndPopulation.py')
    area_num = parse_args()
    curr_area_sweep = areas_sweep[area_num]
    #curr_area_sweep = areas_sweep[2]
    print('GoalI Dims: '+str(curr_area_sweep))
    simple_projection_routine_returns = simple_projection_routine(curr_area_sweep = curr_area_sweep, save_figure = True)
    boot_results_dict = boostrapped_procrustes_projection_routine(curr_area_sweep = curr_area_sweep, master_projection = True)
    EV_plot_returns = EV_GoalDimensions_C2C4_plot(curr_area_sweep = curr_area_sweep, signficance_iteration_samples = 1000)
    manuscripts_plot(results_dict=simple_projection_routine_returns[0], boot_results_dict = boot_results_dict,
        ev_data=EV_plot_returns, plot_lines=goal_numbers, 
        plot_name='Manuscript_{0}'.format(curr_area_sweep[1]), resultsLoc=resultsLoc)
    manuscripts_plot_4step(results_dict=simple_projection_routine_returns[0], boot_results_dict = boot_results_dict,
        ev_data=EV_plot_returns, plot_lines=goal_numbers, 
        plot_name='Manuscript_{0}'.format(curr_area_sweep[1]), resultsLoc=resultsLoc)
    sys.exit()

# Create import_main function to be used in related script
def import_main(curr_area_sweep = areas_sweep[1]):
    simple_projection_routine_returns = simple_projection_routine(curr_area_sweep = curr_area_sweep, save_figure = True)
    EV_plot_returns = EV_GoalDimensions_C2C4_plot(curr_area_sweep = curr_area_sweep, signficance_iteration_samples = 1000)
