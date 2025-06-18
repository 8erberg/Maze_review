
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# File-Name:    dimsSpikeGoalPositionMoveI.py
# Version:      0.0.13 (see end of script for version descriptions)

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
import itertools
from scipy.spatial import distance
from scipy.stats import ttest_ind
from itertools import combinations
from statsmodels.stats.multitest import multipletests

# Import own packages
sys.path.append('...')
from Functions.Import.LocationsI import spikesDataII, resultsFolderI
from Functions.Preprocessing.dictionariesI import arealDicII
from PreAnalysis.fsSpike_GallC_100ms_StimOn_Go_II import anchortimes, import_processed_spike_data, create_time_window_names
from Functions.Preprocessing.MazeMetadataI import mazes

# Locations
resultsLoc = resultsFolderI+"Dimensions/dimsSpikeGoalPositionMoveI/"

# Import results from prior analyses
from Dimensions.dimsSpikeGoalI_2ndPopulation import simple_projection_routine as goalI_projection_routine
from Dimensions.dimsSpikeGoalI_2ndPopulation import areas_sweep, create_data_subselection, parse_args
from Dimensions.dimsSpikeGoalI_2ndPopulation import create_GoalDimensions_C2C4
from Dimensions.dimsSpikeGoalI_2ndPopulation import windows_for_dims_dicts as windows_for_dims_dicts_goalspace
areas_sweep = areas_sweep[:-1]

plot_name_map = {
    'vPFC':'vlPFC',
    'dmPFC':'dmPFC',
    'dlPM':'dPM',
    'Insula':'I/O',
    'PS':'PS',
    'dPFC':'dPFC'}
color_palette = {
    "vlPFC": "#E37A7A",
    "dmPFC": "#6091D6",
    "dPM": "#70A44C",
    "I/O": "#FFC000",
    "PS": "#E7AC84",
    "dPFC": "#D3D3D3"
}

# Windows for PCA projections
grouped_projection_windows = {
        'Goal':('1_all',['TRGDLY_0.1','TRGDLY_0.2','TRGDLY_0.3']),
        'C1':('1_all',['UNFIXTG11_ON_-0.2','UNFIXTG11_ON_-0.1', 'UNFIXTG11_ON_0.0']),
        'C2':('2_2step',['UNFIXTG21_ON_-0.2', 'UNFIXTG21_ON_-0.1', 'UNFIXTG21_ON_0.0']),
        'C3':('3_4step',['UNFIXTG31_ON_-0.2', 'UNFIXTG31_ON_-0.1', 'UNFIXTG31_ON_0.0']),
        'C4':('4_4step',['UNFIXTG41_ON_-0.2', 'UNFIXTG41_ON_-0.1', 'UNFIXTG41_ON_0.0'])}
windows_for_projection_series = pd.Series(grouped_projection_windows.keys())
windows_for_projection = list(grouped_projection_windows.keys())
x_vals = np.array(range(len(windows_for_projection)))
goal_numbers = [7,9,17,19]
goal_locations = goal_numbers
move_direction = ['l','r','u','d']
move_directions = move_direction
FP_to_position = {
    3:'u',8:'u',
    14:'r',15:'r',
    11:'l',12:'l',
    18:'d',23:'d'
}
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
pathvar_options = list(product(goal_locations,move_directions))
pathvar_options = ['_'.join([str(i) for i in i_path]) for i_path in pathvar_options]

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

# Information on goal and move combinations for each choice
combination_dict = {}
for curr_choice in ['ChoiceI', 'ChoiceII', 'ChoiceIII', 'ChoiceIV']:
    goal_move_combinations_woLoc = list(product(goal_locations,move_directions))
    goal_move_combinations = []
    for i_comb in goal_move_combinations_woLoc:
        if curr_choice=='ChoiceII':
            curr_loc = mazes.loc[
                (mazes['ChoiceNo']==curr_choice)&
                (mazes['goal']==i_comb[0])&
                (mazes['NextFPmap']==i_comb[1])&
                (mazes['Nsteps']==2)].FP.unique()           
        else:
            curr_loc = mazes.loc[
                (mazes['ChoiceNo']==curr_choice)&
                (mazes['goal']==i_comb[0])&
                (mazes['NextFPmap']==i_comb[1])].FP.unique()
        if len(curr_loc)==1:
            goal_move_combinations.append((i_comb[0],i_comb[1],curr_loc[0]))
        elif len(curr_loc)==0:
            pass
        else:
            raise ValueError('More than one location found for goal {0} and move {1}'.format(i_comb[0],i_comb[1]))
    combination_dict[curr_choice] = goal_move_combinations

############ Functions ############

def create_path_num_dict(mazes):
    '''
    Create a dictionary of path numbers for each path length
    '''
    path_nums_all = mazes.Npath.unique()
    path_nums_2step = mazes.loc[(mazes['Nsteps']==2)].Npath.unique()
    path_nums_4step = mazes.loc[(mazes['Nsteps']==4)].Npath.unique()
    path_nums_dict = {'all': path_nums_all, '2step':path_nums_2step, '4step':path_nums_4step}
    return path_nums_dict

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

def create_multiindex_dicts(unit_list_input):
    unit_list = unit_list_input.copy()
    unit_list.sort()
    multi_index_dict = {}
    var_combinations = [(7, 'u'),(7, 'l'),(9, 'u'),(9, 'r'),(17, 'l'),(17, 'd'),(19, 'r'),(19, 'd')]
    var_combinations = list(product(unit_list,var_combinations))
    var_combinations = [(i[0],i[1][0],i[1][1]) for i in var_combinations]
    multi_index_dict['default'] = pd.MultiIndex.from_tuples(var_combinations, names=['Unit', 'FinalPos', 'Move'])
    var_combinations = [(7, 'd'),(7, 'r'),(9, 'd'),(9, 'l'),(17, 'r'),(17, 'u'),(19, 'l'),(19, 'u')]
    var_combinations = list(product(unit_list,var_combinations))
    var_combinations = [(i[0],i[1][0],i[1][1]) for i in var_combinations]
    multi_index_dict['4thChoice'] = pd.MultiIndex.from_tuples(var_combinations, names=['Unit', 'FinalPos', 'Move'])
    return multi_index_dict

def create_goalXmove_projection_data(analysisdf, grouped_projection_windows, unit_list):
    '''
    Create data_dict with one df corresponding to the dfs in grouped_projection_windows.
    '''
    multi_index_dict = create_multiindex_dicts(unit_list)
    projection_groups = ['Unit','FinalPos','Move']

    # Create necessary variables
    go_times_list = ['FIXTG0_OFF','FIXTG11_OFF','FIXTG21_OFF','FIXTG31_OFF']
    path_nums_dict = create_path_num_dict(mazes)

    # Get mean activation for time periods
    projection_windows = []
    for key, value in grouped_projection_windows.items():
        analysisdf[key] = analysisdf[value[1]].mean(axis=1)
        projection_windows.append(key)

    # Create data_dict with a key for each df in grouped_projection_windows
    data_dict = {}
    for value in grouped_projection_windows.values():
        data_dict[value[0]] = None

    ## First choice point
    windows_for_projection = ['Goal','C1']
    grouping_data = analysisdf.copy()
    grouping_data = grouping_data.dropna(subset=[go_times_list[0]+'_-0.2'])
    grouping_data['Move'] = grouping_data['Move_c1']
    grouping_data = grouping_data.groupby(projection_groups).mean()[windows_for_projection]
    ### Fill missing values with group mean per Unit
    grouping_data = fill_grouped_data_by_multiindex(grouping_data, multi_index_dict['default'], windows_for_projection)
    grouping_data = grouping_data.reset_index()
    data_dict['1_all']=grouping_data

    ## Second choice point, 2step
    windows_for_projection = ['C2']
    grouping_data = analysisdf.copy()
    grouping_data = grouping_data.dropna(subset=[go_times_list[1]+'_-0.2'])
    grouping_data['Move'] = grouping_data['Move_c2']
    grouping_data = grouping_data.loc[grouping_data['PathNum'].isin(path_nums_dict['2step'])]
    grouping_data = grouping_data.loc[grouping_data['Choice2step']==1]
    grouping_data = grouping_data.groupby(projection_groups).mean()[windows_for_projection]
    ### Fill missing values with group mean per Unit
    grouping_data = fill_grouped_data_by_multiindex(grouping_data, multi_index_dict['default'], windows_for_projection)
    grouping_data = grouping_data.reset_index()
    data_dict['2_2step']=grouping_data

    ## Third choice point
    windows_for_projection = ['C3']
    grouping_data = analysisdf.copy()
    grouping_data = grouping_data.dropna(subset=[go_times_list[2]+'_-0.2'])
    grouping_data['Move'] = grouping_data['Move_c3']
    grouping_data = grouping_data.loc[grouping_data['Choice3step']==1]
    grouping_data = grouping_data.groupby(projection_groups).mean()[windows_for_projection]
    ### Fill missing values with group mean per Unit
    grouping_data = fill_grouped_data_by_multiindex(grouping_data, multi_index_dict['default'], windows_for_projection)
    grouping_data = grouping_data.reset_index()
    data_dict['3_4step']=grouping_data

    ## Combine second and third choice points dataframes
    #windows_for_projection = ['C2','C3']
    #grouping_data = pd.merge(data_dict['2_2step'],data_dict['3_4step'], on=['Unit','FinalPos','Move'], how='outer')
    #grouping_data['C2+C3'] = grouping_data[['C2','C3']].mean(axis=1)
    #grouping_data = grouping_data.drop(columns=['C2','C3'])
    #data_dict['C2+C3']=grouping_data

    ## Fourth choice point
    windows_for_projection = ['C4']
    grouping_data = analysisdf.copy()
    grouping_data = grouping_data.dropna(subset=[go_times_list[3]+'_-0.2'])
    grouping_data['Move'] = grouping_data['Move_c4']
    grouping_data = grouping_data.loc[grouping_data['Choice4step']==1]
    grouping_data = grouping_data.groupby(projection_groups).mean()[windows_for_projection]
    ### Fill missing values with group mean per Unit
    grouping_data = fill_grouped_data_by_multiindex(grouping_data, multi_index_dict['4thChoice'], windows_for_projection)
    grouping_data = grouping_data.reset_index()
    data_dict['4_4step']=grouping_data

    return data_dict

def create_goalXmove_projections(pca_model, data_dict, grouped_projection_windows):
    ## Create dict to hold results
    dim_names = ['dim1','dim2']
    goal_numbers = [7,9,17,19]
    move_directions = ['l','r','u','d']
    results_dict = {}
    results_key_loop = product(dim_names, goal_numbers, move_directions)
    for i_key, i_goal, i_move in results_key_loop:
        results_dict[i_key+'_{0}_{1}'.format(str(i_goal),i_move)]=[]

    ## Project onto dims
    for i_goal in goal_numbers:
        for i_move in move_directions:
            for window_name, window_params in grouped_projection_windows.items():
                grouped_data = data_dict[window_params[0]]
                temp_pop_vector = grouped_data.loc[
                    (grouped_data['FinalPos']==i_goal)&
                    (grouped_data['Move']==i_move)][window_name].values.reshape(1,-1)
                
                # Project onto goal and move dimensions
                for pca_model_dim, i_key in enumerate(dim_names):
                    if temp_pop_vector.shape == (1, 0):
                        results_dict[i_key+'_'+str(i_goal)+'_'+i_move].append(np.nan)
                    else:
                        proj_vec = pca_model.transform(temp_pop_vector).squeeze(0)
                        results_dict[i_key+'_{0}_{1}'.format(str(i_goal),i_move)].append(proj_vec[pca_model_dim])
    
    return results_dict


class ignore_comparison_class:
    '''
    Create element which always returns True for comparison
    '''
    def __eq__(self, other):
        return True

def create_list_of_comparisons(choice_for_combinations, comparison_mode, combination_dict = combination_dict):
    '''
    choice_for_combinations = 'ChoiceI'
    comparison_mode = 'SameGoal&DiffPos'
    combination_dict = combination_dict
    '''
    # Create dict for possible comparisons
    ignore_comparison = ignore_comparison_class()
    map_choiceinfo_to_index = { # Goal, Move, Location
        'SameGoal&DiffMove&DiffPos':(True, False, False),
        'SameMove&DiffGoal&DiffPos':(False, True, False),
        'SameLoc&DiffGoal':(False, ignore_comparison, True),
        'AllDifferent':(False, False, False)}

    # Create list in accordance with comparison criteria
    goal_move_combinations_doubled = list(itertools.combinations(combination_dict[choice_for_combinations],2))
    selected_combinations = []
    for i_comb in goal_move_combinations_doubled:
        comparison_mode_bool = map_choiceinfo_to_index[comparison_mode]    
        observed_comparison_pattern = tuple(i == j for i, j in zip(i_comb[0], i_comb[1]))   
        observed_comparison_mask = tuple(i == j for i, j in zip(observed_comparison_pattern, comparison_mode_bool))
        if all(observed_comparison_mask):
            selected_combinations.append(i_comb)  
    return selected_combinations

def euclidean_distance_over_comparisons(results_dict, 
                                choice_order = ['ChoiceI','ChoiceI','ChoiceII','ChoiceIII','ChoiceIV'],
                                comparison_list = ['SameGoal&DiffMove&DiffPos', 'SameMove&DiffGoal&DiffPos', 'SameLoc&DiffGoal', 'AllDifferent']):
    '''
    Caclulate euclidean distance between two points over lists in results dict.
    '''
    # Create structures to hold results
    distance_list_dict = {}

    # Calculate distances, by comparison, by choice
    for i_comparison in comparison_list:
        distance_list_dict[i_comparison] = []
        for i_num, i_choice in enumerate(choice_order):
            curr_selected_combinations = create_list_of_comparisons(i_choice, i_comparison)
            curr_selected_combinations_distances = []
            for i_comb in curr_selected_combinations:
                point_1_key = '{0}_{1}'.format(i_comb[0][0],i_comb[0][1])
                point_1 = np.array([results_dict['dim1'+'_'+point_1_key][i_num],results_dict['dim2'+'_'+point_1_key][i_num]])
                point_2_key = '{0}_{1}'.format(i_comb[1][0],i_comb[1][1])
                point_2 = np.array([results_dict['dim1'+'_'+point_2_key][i_num],results_dict['dim2'+'_'+point_2_key][i_num]])
                curr_selected_combinations_distances.append(
                    distance.euclidean(point_1,point_2)
                )
            if len(curr_selected_combinations_distances)>0:
                distance_list_dict[i_comparison].append(np.mean(curr_selected_combinations_distances))
            else:
                distance_list_dict[i_comparison].append(np.nan)

    # For each comparison, add mean of 'ChoiceII' and 'ChoiceIII' to the list within distance_list_dict
    for i_comparison in comparison_list:
        choiceII_index = choice_order.index('ChoiceII')
        choiceIII_index = choice_order.index('ChoiceIII')
        distance_list_dict[i_comparison].append(
            np.mean([distance_list_dict[i_comparison][choiceII_index],distance_list_dict[i_comparison][choiceIII_index]])
        )

    return distance_list_dict


def distances_dict_plot(distances_dict, windows_for_projection, curr_area_sweep, resultsLoc):
    distance_result_dict = distances_dict
    windows_for_projection = windows_for_projection
    comparison_list = ['SameGoal&DiffMove&DiffPos', 'SameMove&DiffGoal&DiffPos', 'SameLoc&DiffGoal', 'AllDifferent']

    # Plot results
    fig, ax = plt.subplots()
    for i_comparison in comparison_list:
        #ax.plot(distance_result_dict[i_comparison], label=i_comparison)
        #Alternative way of plotting to allow for NaN values
        tempdistdata = np.array(distance_result_dict[i_comparison])  # Convert to numpy array
        tempdistdata_mask = ~np.isnan(tempdistdata)  # Create mask where True = not NaN
        ax.plot(np.arange(len(tempdistdata))[tempdistdata_mask], tempdistdata[tempdistdata_mask], 'o-', label=i_comparison)
    ax.legend()
    ax.set_title('Distance between combinations, {}'.format(curr_area_sweep[1]))
    # Use windows_for_projection as xticks
    ax.set_xticks(range(len(windows_for_projection)))
    ax.set_xticklabels(windows_for_projection)
    # Save png
    plt.savefig(resultsLoc+'DistancePlot_{}.png'.format(curr_area_sweep[1]),dpi = 300)
    plt.close()


def results_dict_plot(results_dict, windows_for_projection, plot_name, resultsLoc):
    '''
    Create grid plot of projections for all goal periods.
    '''
    # Project each path's mean activity from each window onto each dimension
    pathvar_options = list(product(goal_locations,move_directions))
    pathvar_options = ['_'.join([str(i) for i in i_path]) for i_path in pathvar_options]

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

    # Plot assignment
    plot_specifcations = {
        0:('Goal','mean'),
        1:('C1','mean'),
        2:('C2','split'),
        3:('C3','split'),
        4:('C4','split')
    }
    abbreviations_map = {
        'Goal':'ChoiceI',
        'C1':'ChoiceI',
        'C2':'ChoiceII',
        'C3':'ChoiceIII',
        'C4':'ChoiceIV'
    }

    # Setup figure
    fig, axes = plt.subplots(nrows=1, ncols=len(windows_for_projection), figsize=(6, 2))
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

    # Plot data
    for x_pos, (window_name, window_plot_type) in plot_specifcations.items():
        if window_plot_type == 'split':
            for i_path in pathvar_options:
                # Split pathname
                path_goal, path_move = i_path.split('_')
                path_goal = int(path_goal)
                # Get data
                x_vals = np.array(results_dict['dim1_{}_{}'.format(path_goal,path_move)])
                y_vals = np.array(results_dict['dim2_{}_{}'.format(path_goal,path_move)])

                # Check that neither the x or y values are nan
                if np.isnan(x_vals[x_pos]) or np.isnan(y_vals[x_pos]):
                    continue
                else:
                    # Get the direction for current projection by extracting direction of first move of respective problem
                    curr_ChoiceNo = abbreviations_map[window_name]
                    if curr_ChoiceNo=='ChoiceII':
                        curr_maze_problem = mazes.loc[
                            (mazes['ChoiceNo']==curr_ChoiceNo)&
                            (mazes['goal']==path_goal)&
                            (mazes['NextFPmap']==path_move)&
                            (mazes['Nsteps']==2)].Npath.values[0]
                        curr_first_move = mazes.loc[
                            (mazes['Npath']==curr_maze_problem)&
                            (mazes['ChoiceNo']=='ChoiceI')].NextFPmap.values[0]
                    else:
                        curr_maze_problem = mazes.loc[
                            (mazes['ChoiceNo']==curr_ChoiceNo)&
                            (mazes['goal']==path_goal)&
                            (mazes['NextFPmap']==path_move)].Npath.values[0]
                        curr_first_move = mazes.loc[
                            (mazes['Npath']==curr_maze_problem)&
                            (mazes['ChoiceNo']=='ChoiceI')].NextFPmap.values[0]

                    axes[x_pos].scatter(
                        x_vals[x_pos], y_vals[x_pos], marker=move_markers[curr_first_move], color=goal_colours[path_goal], alpha=1)
        elif window_plot_type == 'mean':
            for i_goal in goal_locations:
                x_vals = []
                y_vals = []
                for i_move in move_directions:
                    x_vals.append(np.array(results_dict['dim1_{}_{}'.format(i_goal,i_move)])[x_pos])
                    y_vals.append(np.array(results_dict['dim2_{}_{}'.format(i_goal,i_move)])[x_pos])
                x_vals_mean = np.nanmean(x_vals)
                y_vals_mean = np.nanmean(y_vals)
                axes[x_pos].scatter(
                            x_vals_mean, y_vals_mean, marker='o', color=goal_colours[i_goal], alpha=1)

    # Add legend
    for i_goal in goal_locations:
        axes[-1].plot([], [],'-', marker='o', color=goal_colours[i_goal], alpha=0.5, label=str(i_goal))
    for i_move in move_directions:
        axes[-1].plot([], [],'-', marker=move_markers[i_move], color='black', alpha=0.5, label=str(i_move))
    axes[-1].legend(loc='center right', bbox_to_anchor=(1.3, 0.5))

    # Add titles to subplots
    for ax, col in zip(axes, windows_for_projection):
        ax.set_title(col, size=BIGGER_SIZE)

    # Add the same x and y axis limits to all subplots within a row, scale them dynamically to min and max across all subplots within a row
    x_min = np.min([ax.get_xlim()[0] for ax in axes])
    x_max = np.max([ax.get_xlim()[1] for ax in axes])
    y_min = np.min([ax.get_ylim()[0] for ax in axes])
    y_max = np.max([ax.get_ylim()[1] for ax in axes])
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Hide the number on x axis for all plots but the plot in the first column
    for ax in axes.flat:
        ax.label_outer()

    # Save figure
    plt.tight_layout()
    # Make font editable in svg
    plt.rcParams['svg.fonttype'] = 'none'
    # Set font to arial
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'arial'
    #plt.savefig(resultsLoc+'print/print_{0}.svg'.format(plot_name), format='svg',dpi=300, bbox_inches="tight")
    plt.savefig(resultsLoc+'{0}.png'.format(plot_name),dpi = 300)
    plt.close()

def apply_fdr_correction(p_values_df, alpha = 0.05):
    # Extract the upper triangular part of the p-values DataFrame
    p_values = p_values_df.values[np.triu_indices_from(p_values_df, k=1)]
    
    # Apply FDR correction
    corrected_p_values = multipletests(p_values, alpha=alpha, method='fdr_bh')[1]
    
    # Create a new DataFrame to store corrected p-values
    corrected_p_values_df = p_values_df.copy()
    
    # Fill the corrected p-values back into the DataFrame
    corrected_p_values_df.values[np.triu_indices_from(corrected_p_values_df, k=1)] = corrected_p_values
    corrected_p_values_df.values[np.tril_indices_from(corrected_p_values_df, -1)] = corrected_p_values_df.values.T[np.tril_indices_from(corrected_p_values_df, k=-1)]
    
    return corrected_p_values_df


if __name__ == "__main__":
    area_num = parse_args()
    #area_num = 2
    curr_area_sweep = areas_sweep[area_num]
    print("Script run: dimsSpikeGoalPositionMoveI.py")
    print('GoalI Dims: '+str(curr_area_sweep))
    #curr_area_sweep = areas_sweep[2]
    #for curr_area_sweep in areas_sweep:
    # Get PCA Space

    ### Pre investigation line
    #_, pca_model, _ = goalI_projection_routine(curr_area_sweep = curr_area_sweep, save_figure = False)
    ### Pre investigation line, end

    # Create projections and distances without bootstrapping
    analysisdf, unit_list, data_area = create_data_subselection(curr_area_sweep, analysisdf_import)
    pca_model, _ = create_GoalDimensions_C2C4(
        analysisdf,
        windows_for_dims_dicts_goalspace, 
        unit_list = unit_list, 
        pca_components=2, 
        return_df = False)
    data_dict = create_goalXmove_projection_data(analysisdf, grouped_projection_windows, unit_list)
    results_dict = create_goalXmove_projections(pca_model, data_dict, grouped_projection_windows)
    distances_dict = euclidean_distance_over_comparisons(results_dict)
    # Create plots
    results_dict_plot(results_dict, windows_for_projection, "Proj_{0}".format(curr_area_sweep[1]), resultsLoc)
    distances_windows_for_projection = windows_for_projection
    distances_windows_for_projection.append('C2+C3')
    distances_dict_plot(distances_dict, distances_windows_for_projection, curr_area_sweep, resultsLoc)


    # Visualise projections at choice 2 and 3 from results_dict by goal and position
    # Make projection figure with two subplots (left: 2nd choice, right: 3rd choice)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2, 1.1))
    for i_choice_str, i_choice  in [('ChoiceII',2),('ChoiceIII',3)]:
        for i_path in pathvar_options:
            # Split pathname
            path_goal, path_move = i_path.split('_')
            path_goal = int(path_goal)
            # Get data
            x_val = results_dict['dim1_{}_{}'.format(path_goal,path_move)][i_choice]
            y_val = results_dict['dim2_{}_{}'.format(path_goal,path_move)][i_choice]

            # Check that neither the x or y values are nan
            if np.isnan(x_val) or np.isnan(y_val):
                continue
            else:
                # Get current position
                if i_choice_str=='ChoiceII':
                    curr_position = mazes.loc[
                        (mazes['ChoiceNo']==i_choice_str)&
                        (mazes['goal']==int(path_goal))&
                        (mazes['NextFPmap']==path_move)&
                        (mazes['Nsteps']==2)].FP.unique()
                elif i_choice_str=='ChoiceIII':
                    curr_position = mazes.loc[
                        (mazes['ChoiceNo']==i_choice_str)&
                        (mazes['goal']==int(path_goal))&
                        (mazes['NextFPmap']==path_move)&
                        (mazes['Nsteps']==4)].FP.unique()
                if len(curr_position)>1:
                    raise ValueError('More than one location found for goal {0} and move {1}'.format(path_goal,path_move))
                curr_position = FP_to_position[curr_position[0]]
                axes[i_choice-2].scatter(
                    x_val, y_val, marker=move_markers[curr_position], color=goal_colours[path_goal], alpha=1, s=20)
        axes[i_choice-2].set_title('Choice {0}'.format(i_choice))

    # Set the same min and max for both axes
    x_min = np.min([axes[i_choice-2].get_xlim()[0] for i_choice in [2,3]])
    x_max = np.max([axes[i_choice-2].get_xlim()[1] for i_choice in [2,3]])
    y_min = np.min([axes[i_choice-2].get_ylim()[0] for i_choice in [2,3]])
    y_max = np.max([axes[i_choice-2].get_ylim()[1] for i_choice in [2,3]])
    global_min = min(x_min, y_min)-0.3
    global_max = max(x_max, y_max)+0.3
    for i_choice in [2,3]:
        axes[i_choice-2].set_xlim(global_min, global_max)
        axes[i_choice-2].set_ylim(global_min, global_max)
        # Set labels for axes
        axes[i_choice-2].set_xlabel('PCA dimension 1')
        axes[i_choice-2].set_ylabel('PCA dimension 2')  
    
    # Save figure
    plt.tight_layout()
    plt.savefig(resultsLoc+'Projection_C2C3_{0}.png'.format(curr_area_sweep[1]),dpi = 300)
    # Save svg
    plt.savefig(resultsLoc+'Projection_C2C3_{0}.svg'.format(curr_area_sweep[1]), format='svg',dpi = 300)
    plt.close()

    ###### Bootstrap distances

    # Create dict with same keys as distance_dict but empty lists as values
    distance_dict_bootstrap = {}
    for i_key in distances_dict.keys():
        distance_dict_bootstrap[i_key] = []

    # Created bootstrapped projections
    analysisdf.reset_index(inplace=True)
    #stratification_var = analysisdf['Unit'].astype(str)
    results_dict_boot_list = []
    for i_split in range(1000):
        # Resample data (now by unit)
        #sampled_index = resample(analysisdf.index, replace=True, n_samples=len(analysisdf), stratify=stratification_var)
        unit_list = analysisdf['Unit'].unique()
        unit_list = resample(unit_list,replace=True, n_samples=len(unit_list))
        analysisdf_boot = []
        for i_num, i_unit in enumerate(unit_list):
            appended_df = analysisdf.loc[analysisdf['Unit']==i_unit]
            appended_df['Unit'] = 'Unit_'+str(i_num)
            analysisdf_boot.append(appended_df)
        analysisdf_boot = pd.concat(analysisdf_boot)
        unit_list = analysisdf_boot['Unit'].unique()
        unit_list.sort()
        print(f"Fold {i_split}:")
        #analysisdf_boot = analysisdf.iloc[sampled_index]
        # Get projections from sampled data
        pca_model, _ = create_GoalDimensions_C2C4(
            analysisdf_boot,
            windows_for_dims_dicts_goalspace, 
            unit_list = unit_list, 
            pca_components=2, 
            return_df = False)
        data_dict_boot = create_goalXmove_projection_data(analysisdf_boot, grouped_projection_windows, unit_list)
        results_dict_boot = create_goalXmove_projections(pca_model, data_dict_boot, grouped_projection_windows)
        results_dict_boot_list.append(results_dict_boot)
        distances_dict_boot = euclidean_distance_over_comparisons(results_dict_boot)

        ## Add each list from results_dict to boot_results_dict
        for key, value in distances_dict_boot.items():
            distance_dict_bootstrap[key].append(value)


    # Turn lists into arrays
    for key, value in distance_dict_bootstrap.items():
        distance_dict_bootstrap[key] = np.array(value)

    # Create histogram plot based on distance_dict_bootstrap
    histogram_arrays = {}
    for key, value in distance_dict_bootstrap.items():
        if key == 'SameMove&DiffGoal&DiffPos':
            pass
        else:
            histogram_arrays[key] = value[:,-1]
    fig, ax = plt.subplots()
    # Create color scheme for each key
    colors = plt.cm.viridis(np.linspace(0,1,len(histogram_arrays.keys())))
    colors = [matplotlib.colors.rgb2hex(i) for i in colors]
    color_map = dict(zip(histogram_arrays.keys(),colors))

    for key, value in histogram_arrays.items():
        # Add histogram and mean line in same color
        ax.hist(value, bins=20, alpha=0.5, label=key, color=color_map[key])
        ax.axvline(value.mean(), color=color_map[key], linestyle='dashed', linewidth=1)

    ax.legend()
    ax.set_title('{0} - Histogram of distances for bootstrapped samples'.format(curr_area_sweep[1]))
    plt.savefig(resultsLoc+'Histogram_{0}.png'.format(curr_area_sweep[1]),dpi = 300)
    plt.close()

    # Calculate p-values for selected distances and windows, based on RankTest
    distances_for_stats = list(distance_dict_bootstrap.keys())
    windows_for_stats = [2,3,5]
    stats_results_dict = {}
    for i_window in windows_for_stats:
        stats_results_dict[i_window] = pd.DataFrame(index=distances_for_stats, columns=distances_for_stats)
        for (key1, key2) in combinations(distances_for_stats, 2):
            # Get difference from original distance_dict
            baseline_val = distances_dict[key1][i_window]-distances_dict[key2][i_window]

            # Get array of observations under bootstrap
            data_array1 = distance_dict_bootstrap[key1][:,i_window]
            data_array2 = distance_dict_bootstrap[key2][:,i_window]
            diff_array = data_array1-data_array2

            # Get p-value
            #if baseline_val>0:
            #    p_value = np.sum(diff_array<0)/len(diff_array)
            #elif baseline_val<0:
            #    p_value = np.sum(diff_array>0)/len(diff_array)
            #else:
            #    raise ValueError('Baseline value is 0')
            #if p_value>0.5:
            #    p_value = 1-p_value
            #p_value = p_value*2

            # When baseline is 0, there should be no significant difference
            if baseline_val == 0:
                p_value = 1.0  # Not significant
            else:
                # Calculate p-value based on direction of effect
                if baseline_val > 0:
                    p_value = np.sum(diff_array < 0)/len(diff_array)
                else:  # baseline_val < 0
                    p_value = np.sum(diff_array > 0)/len(diff_array)
                
                # Convert to two-tailed p-value
                if p_value > 0.5:
                    p_value = 1 - p_value
                p_value = p_value * 2

            stats_results_dict[i_window].loc[key1,key2] = p_value
            stats_results_dict[i_window].loc[key2,key1] = p_value
        # Save the df to disk
        stats_results_dict[i_window].to_csv(resultsLoc+'stats_results_dict_{0}_{1}.csv'.format(curr_area_sweep[1],i_window))


    # Apply FDR correction to the p-values & save the df to disk
    stats_results_dict_fdr = {}
    for alpha_val in [0.05, 0.01]:
        for i_window in windows_for_stats:
            stats_results_dict_fdr[i_window] = apply_fdr_correction(stats_results_dict[i_window], alpha = alpha_val)
            stats_results_dict_fdr[i_window].to_csv(resultsLoc+'stats_results_dict_{0}_{1}_alpha{2}.csv'.format(curr_area_sweep[1],i_window, str(alpha_val)))

    # Build bar chart distances_dict (only using the last value of each list), with one bar for each comparison. Highlight p values of significant group comparisons based on stats_results_dict_fdr (0.05 values)
    area_plot_name = plot_name_map[curr_area_sweep[1]]
    curr_bar_color = color_palette[area_plot_name]
    
    # Provided dict for plotting
    distances_for_plot = {
        'SameGoal&DiffMove&DiffPos': 'Same Goal',
        'SameLoc&DiffGoal': 'Same Location', 
        'AllDifferent': 'Baseline'
    }

    # Function to determine significance stars
    def get_significance_label(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''

    # Filter and prepare bar chart values based on distances_for_plot
    bar_chart_values = {key: value[-1] for key, value in distances_dict.items() if key in distances_for_plot}
    bar_chart_values = pd.Series(bar_chart_values).reindex(distances_for_plot.keys())

    # Create font information 
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 6,6,8
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    # Create the bar chart
    fig, ax = plt.subplots(figsize=(3, 2)) # Make the plot smaller
    bars = bar_chart_values.plot(kind='bar', ax=ax, color=curr_bar_color)

    # Rotate labels by 45 degrees
    ax.set_xticklabels([distances_for_plot[key] for key in bar_chart_values.index], rotation=0)

    # Add y-axis label
    ax.set_ylabel('Distance')

    # Calculate spacing for significance bars
    y_max = bar_chart_values.max()
    bar_height_increment = y_max * 0.08  # Space between significance bars
    used_y_positions = []  # Keep track of used y-positions

    # Highlight significant values
    for i_key1, i_key2 in combinations(distances_for_plot.keys(), 2):
        if stats_results_dict[5].loc[i_key1, i_key2] < 0.05:
            x1, x2 = bar_chart_values.index.get_loc(i_key1), bar_chart_values.index.get_loc(i_key2)
            
            # Find the next available y position
            y_base = max(bar_chart_values[i_key1], bar_chart_values[i_key2])
            y = y_base + bar_height_increment
            
            # Ensure we don't overlap with existing bars
            while any(abs(y - used_y) < bar_height_increment/2 for used_y in used_y_positions):
                y += bar_height_increment
            
            used_y_positions.append(y)
            
            # Draw the significance bar
            ax.plot([x1, x1], [bar_chart_values[i_key1], y], 'k-', lw=1.5)
            ax.plot([x1, x2], [y, y], 'k-', lw=1.5)
            ax.plot([x2, x2], [y, bar_chart_values[i_key2]], 'k-', lw=1.5)
            
            # Add significance stars
            significance_label = get_significance_label(stats_results_dict[5].loc[i_key1, i_key2])
            ax.text((x1 + x2) * 0.5, y + 0.02, significance_label, ha='center', va='bottom', color='black')

    # Adjust y-axis limit to accommodate significance bars
    if used_y_positions:  # Only adjust if we have significance bars
        ax.set_ylim(0, max(used_y_positions) + bar_height_increment)

    ax.set_title('{} - Bar chart of distances'.format(area_plot_name))

    # Use tight_layout to avoid cutting off text
    plt.tight_layout()

    # Make font editable in svg
    plt.rcParams['svg.fonttype'] = 'none'

    # Remove top and right spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the figure as .png and .svg
    plt.savefig(resultsLoc + 'BarChart_{0}.png'.format(curr_area_sweep[1]), dpi=300)
    plt.savefig(resultsLoc + 'BarChart_{0}.svg'.format(curr_area_sweep[1]), bbox_inches='tight')
    plt.close()

    # Using bootstrapped results, visualise projections at choice 2 and 3 from results_dict by goal and position
    # Make projection figure with two subplots (left: 2nd choice, right: 3rd choice)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2, 1.1))
    for i_choice_str, i_choice  in [('ChoiceII',2),('ChoiceIII',3)]:
        for i_path in pathvar_options:
            # Split pathname
            path_goal, path_move = i_path.split('_')
            path_goal = int(path_goal)
            x_val_list = []
            y_val_list = []
            for i_boot_results in results_dict_boot_list:
                # Get data
                x_val_temp = i_boot_results['dim1_{}_{}'.format(path_goal,path_move)][i_choice]
                y_val_temp = i_boot_results['dim2_{}_{}'.format(path_goal,path_move)][i_choice]
                # Check that neither the x or y values are nan
                if np.isnan(x_val_temp) or np.isnan(y_val_temp):
                    continue
                else:
                    x_val_list.append(x_val_temp)
                    y_val_list.append(y_val_temp)
            if len(x_val_list)==0:
                continue
            elif len(x_val_list)==len(results_dict_boot_list):
                # Get mean across bootstraps
                x_val = np.nanmean(x_val_list)
                y_val = np.nanmean(y_val_list)
                # Get current position
                if i_choice_str=='ChoiceII':
                    curr_position = mazes.loc[
                        (mazes['ChoiceNo']==i_choice_str)&
                        (mazes['goal']==int(path_goal))&
                        (mazes['NextFPmap']==path_move)&
                        (mazes['Nsteps']==2)].FP.unique()
                elif i_choice_str=='ChoiceIII':
                    curr_position = mazes.loc[
                        (mazes['ChoiceNo']==i_choice_str)&
                        (mazes['goal']==int(path_goal))&
                        (mazes['NextFPmap']==path_move)&
                        (mazes['Nsteps']==4)].FP.unique()
                if len(curr_position)>1:
                    raise ValueError('More than one location found for goal {0} and move {1}'.format(path_goal,path_move))
                curr_position = FP_to_position[curr_position[0]]
                axes[i_choice-2].scatter(
                    x_val, y_val, marker=move_markers[curr_position], color=goal_colours[path_goal], alpha=1, s=15)
            else:
                raise ValueError('Inconsistent bootstrapped results')
        axes[i_choice-2].set_title('Choice {0}'.format(i_choice))

    # Set the same min and max for both axes
    x_min = np.min([axes[i_choice-2].get_xlim()[0] for i_choice in [2,3]])
    x_max = np.max([axes[i_choice-2].get_xlim()[1] for i_choice in [2,3]])
    y_min = np.min([axes[i_choice-2].get_ylim()[0] for i_choice in [2,3]])
    y_max = np.max([axes[i_choice-2].get_ylim()[1] for i_choice in [2,3]])
    global_min = min(x_min, y_min)-0.3
    global_max = max(x_max, y_max)+0.3
    for i_choice in [2,3]:
        axes[i_choice-2].set_xlim(global_min, global_max)
        axes[i_choice-2].set_ylim(global_min, global_max)
        # Set labels for axes
        axes[i_choice-2].set_xlabel('PCA dimension 1')
        axes[i_choice-2].set_ylabel('PCA dimension 2')  
    
    # Save figure
    plt.tight_layout()
    plt.savefig(resultsLoc+'Projection_C2C3_{0}_boot.png'.format(curr_area_sweep[1]),dpi = 300)
    # Save svg
    plt.savefig(resultsLoc+'Projection_C2C3_{0}_boot.svg'.format(curr_area_sweep[1]), format='svg',dpi = 300)
    plt.close()
