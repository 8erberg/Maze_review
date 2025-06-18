#!/usr/bin/env python
# -*- coding: utf-8 -*-

# File-Name:    dimsSpikeC1II_4step.py
# Version:      0.0.5 (see end of script for version descriptions)

############ Setup ############
# Import packages
import sys
import pandas as pd
pd.options.mode.chained_assignment = None # Turn off warnings for chained assignments
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib
from itertools import product
import itertools
from sklearn.utils import resample
from scipy.spatial import distance
from statsmodels.stats.multitest import multipletests

# Import own packages
sys.path.append('...')
from Functions.Import.LocationsI import spikesDataII, resultsFolderI
from Functions.Preprocessing.dictionariesI import arealDicII
from PreAnalysis.fsSpike_GC1C2_50ms_StimOn_Go_I import anchortimes, import_processed_spike_data, create_time_window_names
from Functions.Preprocessing.MazeMetadataI import mazes
from Dimensions.dimsSpikeGoalI_2ndPopulation import create_path_num_dict

# Import results from prior analyses
from Dimensions.dimsSpikeGoalI_2ndPopulation import simple_projection_routine as goalI_projection_routine
from Dimensions.dimsSpikeMoveI_2ndPopulation import simple_projection_routine as moveI_projection_routine
from Dimensions.dimsSpikeGoalI_2ndPopulation import areas_sweep, create_data_subselection, parse_args
from Dimensions.dimsSpikeMoveI_2ndPopulation import create_MoveDimensions_C2C4
from Dimensions.dimsSpikeMoveI_2ndPopulation import windows_for_dims_dicts as windows_for_dims_dicts_move
# We need to recode the window names for the move dimensions as we are extracting different windows
windows_for_dims_dicts_move = {
    'C2': ['UNFIXTG21_ON_0.2', 'UNFIXTG21_ON_0.25', 'UNFIXTG21_ON_0.3', 'UNFIXTG21_ON_0.35'], 
    'C4': ['UNFIXTG41_ON_0.2', 'UNFIXTG41_ON_0.25', 'UNFIXTG41_ON_0.3', 'UNFIXTG41_ON_0.35']}
pca_spaces_dict = {
    'movec24':moveI_projection_routine
}
path_nums_dict = create_path_num_dict(mazes)

# Locations
resultsLoc = resultsFolderI+"Dimensions/dimsSpikeC1II_4step/"

# Import data
analysisdf_import = import_processed_spike_data()
analysisdf_import['og_area'] = analysisdf_import['area']
analysisdf_import['Unit'] = analysisdf_import['session']+'_'+analysisdf_import['NeuronNum'].astype(str)

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

windows_for_projection = ['UNFIXTG21_ON_-0.4',
       'UNFIXTG21_ON_-0.35', 'UNFIXTG21_ON_-0.3', 'UNFIXTG21_ON_-0.25',
       'UNFIXTG21_ON_-0.2', 'UNFIXTG21_ON_-0.15', 'UNFIXTG21_ON_-0.1',
       'UNFIXTG21_ON_-0.05', 'UNFIXTG21_ON_0.0', 'UNFIXTG21_ON_0.05',
        'UNFIXTG21_ON_0.1', 'UNFIXTG21_ON_0.15', 'UNFIXTG21_ON_0.2',
       'UNFIXTG21_ON_0.25', 'UNFIXTG21_ON_0.3', 'UNFIXTG21_ON_0.35',
       'FIXTG11_OFF_-0.2', 'FIXTG11_OFF_-0.15', 'FIXTG11_OFF_-0.1',
       'FIXTG11_OFF_-0.05', 'FIXTG11_OFF_0.0', 'FIXTG11_OFF_0.05']
windows_for_projection_plot = ['Ch2_-0.4',
       'Ch2_-0.35', 'Ch2_-0.3', 'Ch2_-0.25',
       'Ch2_-0.2', 'Ch2_-0.15', 'Ch2_-0.1',
       'Ch2_-0.05', 'Ch2_0.0', 'Ch2_0.05',
        'Ch2_0.1', 'Ch2_0.15', 'Ch2_0.2',
        'Ch2_0.25', 'Ch2_0.3', 'Ch2_0.35',
       'Go2_-0.2', 'Go2_-0.15', 'Go2_-0.1',
       'Go2_-0.05', 'Go2_0.0', 'Go2_0.05']

# Project each path's mean activity from each window onto each dimension
pathvar_options = list(product(goal_locations,move_directions))
pathvar_options = ['_'.join([str(i) for i in i_path]) for i_path in pathvar_options]

############ Functions ############
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-area_num", type=int, 
                        help="Area for analysis, chosen as numerical from list")
    args = parser.parse_args()
    area_num = args.area_num
    return area_num

def create_projection_data(analysisdf, group_by_move, windows_for_projection, pathnum_grouping = False):
    '''
    Project set of goalperiod windows on all pca_models (first two dims) provided.
    '''
    grouped_data = analysisdf.copy()
    unit_list = grouped_data['Unit'].unique()
    unit_list.sort()

    # Prepare projection data
    ## First choice point
    grouped_data = grouped_data.loc[(grouped_data['Choice2step']==1)].dropna(subset=['FIXTG11_OFF_0.05'])
    # Select only 2 or 4 step trials if needed
    if pathnum_grouping != False:
        curr_pathnums = path_nums_dict[pathnum_grouping]
        grouped_data = grouped_data.loc[grouped_data['PathNum'].isin(curr_pathnums)]
    if group_by_move:
        # Create grouping
        grouped_data = grouped_data.groupby(['Unit','FinalPos','Move_c1']).mean()[windows_for_projection]
        # Create version of full index
        var_combinations = [(7, 'u'),(7, 'l'),(9, 'u'),(9, 'r'),(17, 'l'),(17, 'd'),(19, 'r'),(19, 'd')]
        var_combinations = list(product(unit_list,var_combinations))
        var_combinations = [(i[0],i[1][0],i[1][1]) for i in var_combinations]
        multi_index = pd.MultiIndex.from_tuples(var_combinations, names=['Unit', 'FinalPos', 'Move'])
    else:
        # Create grouping
        grouped_data = grouped_data.groupby(['Unit','FinalPos']).mean()[windows_for_projection].reset_index()
        # Create version of full index
        var_combinations = [7,9,17,19]
        var_combinations = list(product(unit_list,var_combinations))
        var_combinations = [(i[0],i[1]) for i in var_combinations]
        multi_index = pd.MultiIndex.from_tuples(var_combinations, names=['Unit', 'FinalPos'])
    # Update index to full index
    grouped_data = grouped_data.reindex(multi_index)
    # Fill missing values with group mean per Unit
    means = grouped_data.groupby(['Unit'])[windows_for_projection].transform('mean')
    grouped_data[windows_for_projection] = grouped_data[windows_for_projection].fillna(means)
    
    return grouped_data

def create_goalXmove_projections(pca_dict, grouped_data, goal_locations, move_directions, windows_for_projection):
    ## Create dict to hold results
    dim_names = list(itertools.product(list(pca_dict.keys()),['dim1','dim2']))
    dim_names = [i[0]+'_'+i[1] for i in dim_names]
    results_dict = {}
    results_key_loop = product(dim_names, goal_locations, move_directions)
    for i_key, i_goal, i_move in results_key_loop:
        results_dict[i_key+'_{0}_{1}'.format(str(i_goal),i_move)]=[]

    ## Project onto dims
    for i_goal in goal_locations:
        for i_move in move_directions:
            for window_name in windows_for_projection:
                temp_pop_vector = grouped_data.loc[
                    (grouped_data['FinalPos']==i_goal)&
                    (grouped_data['Move']==i_move)][window_name].values.reshape(1,-1)
                
                # Project onto goal and move dimensions
                for i_num, i_key in enumerate(dim_names):
                    if temp_pop_vector.shape == (1, 0):
                        results_dict[i_key+'_'+str(i_goal)+'_'+i_move].append(np.nan)
                    else:
                        pca_model_name = i_key.split('_')[0]
                        pca_model_dim = int(i_key.split('_')[1][-1])-1
                        pca_model = pca_dict[pca_model_name]
                        proj_vec = pca_model.transform(temp_pop_vector).squeeze(0)
                        results_dict[i_key+'_{0}_{1}'.format(str(i_goal),i_move)].append(proj_vec[pca_model_dim])
    
    return results_dict


############ Execution calls ############

def simple_projection_routine(curr_area_sweep, analysisdf_import, save_figure = False, resample_units = False, pathnum_grouping = None):
    '''
    Basic routine to create projection plots.
    # Apply loop
    for i_area_sweep in areas_sweep:
        _ = simple_projection_routine(i_area_sweep, analysisdf_import, save_figure = True)
    '''
    analysisdf = create_data_subselection(curr_area_sweep, analysisdf_import)[0]
    unit_list = analysisdf['Unit'].unique()
    unit_list.sort()

    if resample_units:
        # Sample set of Units from analysisdf, with replacement
        unit_list = analysisdf['Unit'].unique()
        unit_list = resample(unit_list,replace=True, n_samples=len(unit_list))
        new_analysisdf = []
        for i_num, i_unit in enumerate(unit_list):
            appended_df = analysisdf.loc[analysisdf['Unit']==i_unit]
            appended_df['Unit'] = 'Unit_'+str(i_num)
            new_analysisdf.append(appended_df)
        analysisdf = pd.concat(new_analysisdf)
        unit_list = analysisdf['Unit'].unique()
        unit_list.sort()

    # Get updated PCA space
    pca, pca_data = create_MoveDimensions_C2C4(analysisdf,windows_for_dims_dicts_move, unit_list = unit_list, pca_components=3, return_df = True)
    pca_dict = {'movec24':pca}
    pca_data_dict = {'movec24':pca_data}

    projection_data = create_projection_data(analysisdf, group_by_move=True,
        windows_for_projection=windows_for_projection, pathnum_grouping = pathnum_grouping)
    projection_data = projection_data.reset_index()
    results_dict = create_goalXmove_projections(pca_dict, projection_data, goal_locations, move_directions, windows_for_projection)
    if save_figure:
        sys.exit('No plotting function.')
    return results_dict, pca_dict, pca_data_dict

def get_move_distances(curr_area_sweep, results_dict, pca_dict, pca_data_dict, save_figure = False, figure_name_group = None):
    '''
    Get distances to master move for each move.
    # Apply loop
    for i_area_sweep in areas_sweep:
        get_move_distances(i_area_sweep, analysisdf_import)
    '''
    choice1_distances_dict = {}
    choice2_distances_dict = {}
    non_move_distances_dict = {}


    valid_var_combinations = [(7, 'u'),(7, 'l'),(9, 'u'),(9, 'r'),(17, 'l'),(17, 'd'),(19, 'r'),(19, 'd')]
    goal_to_choices_map = {7:['u','l'],9:['u','r'],17:['l','d'],19:['r','d']}

    # Get master projection coordinates for each move, projection original pca data into the pca space
    master_projections = {}
    for i_move in move_directions:
        temp_data = pca_data_dict['movec24'].reset_index()
        temp_data = temp_data.loc[temp_data['Move']==i_move]['dim_window'].values.reshape(1,-1)
        master_projections[i_move] = pca_dict['movec24'].transform(temp_data).squeeze(0)[0:2]

    # Get distances from respective first and second
    distance_result_dict = {}
    for i_goalchoice in valid_var_combinations:

        dim1 = results_dict['movec24_dim1_{}_{}'.format(i_goalchoice[0],i_goalchoice[1])]
        dim2 = results_dict['movec24_dim2_{}_{}'.format(i_goalchoice[0],i_goalchoice[1])]

        choice1_distances = []
        choice2_distances = []
        non_move_distances = []
        for i_timepoints in range(len(dim1)):
            # Get current observation point
            point_observation = [dim1[i_timepoints],dim2[i_timepoints]]
            
            # Get move of choice 1
            point_master_projections_choice1 = master_projections[i_goalchoice[1]]

            # Get move of choice 2
            move2 = goal_to_choices_map[i_goalchoice[0]].copy()
            move2.remove(i_goalchoice[1])
            move2 = move2[0]
            point_master_projections_choice2 = master_projections[move2]

            # Get non move directions
            non_move_directions = move_directions.copy()
            non_move_directions.remove(i_goalchoice[1])
            non_move_directions.remove(move2)

            # Get distances for single moves
            choice1_distances.append(distance.euclidean(point_observation,point_master_projections_choice1))
            choice2_distances.append(distance.euclidean(point_observation,point_master_projections_choice2))

            # Get mean distance for non move directions
            non_move_distances_templist = []
            for i_non_move in non_move_directions:
                non_move_distances_templist.append(distance.euclidean(point_observation,master_projections[i_non_move]))
            non_move_distances.append(np.mean(non_move_distances_templist))

        # Add lists to dicts
        choice1_distances_dict['{}_{}'.format(i_goalchoice[0],i_goalchoice[1])] = choice1_distances
        choice2_distances_dict['{}_{}'.format(i_goalchoice[0],i_goalchoice[1])] = choice2_distances
        non_move_distances_dict['{}_{}'.format(i_goalchoice[0],i_goalchoice[1])] = non_move_distances

    # Combine all lists of dict into array
    distance_result_dict['choice1'] = np.array(list(choice1_distances_dict.values())).mean(0)
    distance_result_dict['choice2'] = np.array(list(choice2_distances_dict.values())).mean(0)
    distance_result_dict['non_move'] = np.array(list(non_move_distances_dict.values())).mean(0)

    # Plot results
    if save_figure:
        fig, ax = plt.subplots()
        ax.plot(distance_result_dict['choice1'], label='Choice1')
        ax.plot(distance_result_dict['choice2'], label='Choice2')
        ax.plot(distance_result_dict['non_move'], label='NonMove')
        ax.legend()
        ax.set_title('Distance to master move, {}'.format(curr_area_sweep[1]))
        # Use windows_for_projection as xticks
        ax.set_xticks(range(len(windows_for_projection)))
        ax.set_xticklabels(windows_for_projection_plot, rotation = 45)
        # Save png
        plt.savefig(resultsLoc+'DistanceToMasterMove_{}_{}.png'.format(curr_area_sweep[1],figure_name_group),dpi = 300)
        plt.close()

    return distance_result_dict


if __name__ == "__main__":
    print('dimsSpikeC1II_4step.py')
    area_num = parse_args()
    curr_area_sweep = areas_sweep[area_num]
    i_pathgroup = '4step'
    figure_name_group = i_pathgroup
    ### Get master contrasts and main plots
    areas_sweep_manuscript = [ (['Insula'], 'Insula'),
        (['vPFC'], 'vPFC'),
        (['dmPFC'], 'dmPFC'),
        (['dlPM'], 'dlPM')]
    i_area_sweep = areas_sweep_manuscript[area_num]
    curr_area_sweep = areas_sweep_manuscript[area_num]
    across_areas_dict = {}

    # Get baseline projections
    results_dict, pca_dict, pca_data_dict = simple_projection_routine(i_area_sweep, analysisdf_import, save_figure = False, pathnum_grouping = figure_name_group)
    distance_result_dict = get_move_distances(i_area_sweep, results_dict, pca_dict, pca_data_dict, save_figure = True, figure_name_group = figure_name_group)
    
    # Create contrast values in distance_result_dict
    distance_result_dict['choice1_/_non_move'] = distance_result_dict['choice1']/distance_result_dict['non_move']
    distance_result_dict['choice2_/_non_move'] = distance_result_dict['choice2']/distance_result_dict['non_move']
    distance_result_dict['choice1-non_move'] = distance_result_dict['choice1']-distance_result_dict['non_move']
    distance_result_dict['choice2-non_move'] = distance_result_dict['choice2']-distance_result_dict['non_move']
    across_areas_dict[i_area_sweep[1]] = distance_result_dict

    # Get bootstrapped p-values
    distance_result_dict_list = []
    for i_count in range(1000):
        results_dict, pca_dict, pca_data_dict = simple_projection_routine(i_area_sweep, analysisdf_import, save_figure = False, resample_units = True, pathnum_grouping = figure_name_group)
        distance_result_dict = get_move_distances(i_area_sweep, results_dict, pca_dict, pca_data_dict, save_figure = False, figure_name_group = figure_name_group)
        # Create contrast values in distance_result_dict
        distance_result_dict['choice1-non_move'] = distance_result_dict['choice1']-distance_result_dict['non_move']
        distance_result_dict['choice2-non_move'] = distance_result_dict['choice2']-distance_result_dict['non_move']
        distance_result_dict_list.append(distance_result_dict)
        print(i_count)

    # Get bootsrapped p-values
    stats_results_df = pd.DataFrame(index = windows_for_projection_plot, columns = ['choice1-non_move','choice2-non_move'])
    for i_window in range(len(windows_for_projection)):
        for i_contrast in ['choice1-non_move','choice2-non_move']:
            baseline_val = across_areas_dict[curr_area_sweep[1]][i_contrast][i_window]
            diff_array = np.array([i_dict[i_contrast][i_window] for i_dict in distance_result_dict_list])
            ### This is the old way of calculating p-values, but it has bug because of the comparison with 0
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

            stats_results_df.loc[windows_for_projection_plot[i_window],i_contrast] = p_value
    # Apply FDR correction to each column for create columns with corrected p values
    for i_contrast in ['choice1-non_move','choice2-non_move']:
        stats_results_df[i_contrast+'_FDR_0-05'] = multipletests(stats_results_df[i_contrast], method='fdr_bh')[1]
        stats_results_df[i_contrast+'_FDR_0-01'] = multipletests(stats_results_df[i_contrast], method='fdr_bh', alpha=0.01)[1]
    
    # Save to disk
    stats_results_df.to_csv(resultsLoc+'StatsResults_{}.csv'.format(curr_area_sweep[1]))

    # Make plot with signifiance data
    curr_time_data = across_areas_dict[curr_area_sweep[1]]

    group_name_dict = {'choice1':'Chosen','choice2':'Planned','non_move':'Baseline'}
    # Map group names to colors from viridis color scale
    cmap = matplotlib.cm.get_cmap('viridis')
    group_colors = [cmap(i) for i in [0.1,0.5,0.9]]
    group_colors_dict = dict(zip(list(group_name_dict.keys()),group_colors))
    # replace 'non_move' with dark grey
    group_colors_dict['non_move'] = 'grey'

    # Transform ticks
    reference_names = ['Choice 2', 'Go 2']
    reference_names_counter = 0
    new_ticks = []
    for i_tick in windows_for_projection_plot:
        if i_tick.endswith('0.0'):
            new_ticks.append(reference_names[reference_names_counter])
            reference_names_counter+=1
        elif '5' in i_tick:
            new_ticks.append('')
        else:
            new_ticks.append(i_tick.split('_')[1])

    # Get all indices in windows_for_projection_plot starting with Ch1, Go1, Ch2
    index_dict = {}
    for i_key_start in ['Ch2','Go2',]:
        temp_key_list = [i_key for i_key in windows_for_projection_plot if i_key.startswith(i_key_start)]
        index_dict[i_key_start] = [windows_for_projection_plot.index(i_key) for i_key in temp_key_list]

    # Add gaps between periods
    windows_for_projection_plot_extended = []
    new_ticks_extended = []
    last_index = 0

    for period in ['Ch2','Go2',]:
        curr_indices = index_dict[period]
        # Add current period indices and ticks
        while last_index < curr_indices[-1] + 1:
            windows_for_projection_plot_extended.append(windows_for_projection_plot[last_index])
            new_ticks_extended.append(new_ticks[last_index])
            last_index += 1
        # Add empty slot after period (except last period)
        if period != 'Go2':
            windows_for_projection_plot_extended.append('gap')
            new_ticks_extended.append('')

    # Get smallest y values across groups to use as significance line height
    curr_minimum = 100000
    curr_maximum = -100000
    for i_group in ['choice1','choice2','non_move']:
        y_data = curr_time_data[i_group]
        if min(y_data)<curr_minimum:
            curr_minimum = min(y_data)
        if max(y_data)>curr_maximum:
            curr_maximum = max(y_data)
    curr_range = curr_maximum-curr_minimum
        

    # Create plot with with data from Ch1, Go1, Ch2 plotted as unnconnected lines
    fig, ax = plt.subplots(figsize=(3, 2))
    for i_group in ['choice1','choice2','non_move']:
        for i_key_start in ['Ch2','Go2']:
            y_data = curr_time_data[i_group][index_dict[i_key_start]]
            x_data = [i for i, val in enumerate(windows_for_projection_plot_extended) 
                        if val.startswith(i_key_start)]
            x_data = [i + 0.5 for i in x_data]
            ax.plot(x_data,y_data,label=group_name_dict[i_group],color=group_colors_dict[i_group])
            
            # Add significance data, old routine, without offset
            #if i_group in ['choice1','choice2']:
            #    i_sig_num = int(i_group[-1])
            #    p_values_array = stats_results_df[i_group+'-non_move_FDR_0-05'].values
            #    curr_indices = index_dict[i_key_start]
            #    curr_y_height = curr_minimum-curr_range*0.1*i_sig_num
            #    for i_num in curr_indices:
            #        if p_values_array[i_num]<0.05:
            #            ax.plot(i_num+0.5,curr_y_height,color=group_colors_dict[i_group],marker='o',markersize=3)
            
            # Add significance data with correct offset
            if i_group in ['choice1','choice2']:
                i_sig_num = int(i_group[-1])
                p_values_array = stats_results_df[i_group+'-non_move_FDR_0-05'].values
                curr_indices = index_dict[i_key_start]
                curr_y_height = curr_minimum-curr_range*0.1*i_sig_num
                
                # Count how many gaps come before current period
                gap_offset = 0
                if i_key_start == 'Go2':  # Only Go2 needs offset since it comes after the gap
                    gap_offset = 1
                
                for i_num in curr_indices:
                    if p_values_array[i_num]<0.05:
                        ax.plot(i_num+0.5+gap_offset, curr_y_height, 
                            color=group_colors_dict[i_group], marker='o', markersize=3)

    # Only show ticks at data points (exclude gaps)
    tick_positions = []
    tick_labels = []
    # Loop through and only add non-gap positions/labels
    for i, val in enumerate(windows_for_projection_plot_extended):
        if val != 'gap':
            tick_positions.append(i)
            tick_labels.append(new_ticks_extended[i])

    ax.set_title('Distance to master move, {}'.format(curr_area_sweep[1]))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90)
    # Add legend but with only one line per group
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.rcParams['svg.fonttype'] = 'none'
    # Save png
    plt.tight_layout()
    plt.savefig(resultsLoc+'DistanceToMasterMove_{}_{}_Significance.png'.format(curr_area_sweep[1], figure_name_group),dpi = 300)
    plt.savefig(resultsLoc+'DistanceToMasterMove_{}_{}_Significance.svg'.format(curr_area_sweep[1], figure_name_group), bbox_inches="tight")
    plt.close()

