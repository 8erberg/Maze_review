# -*- coding: utf-8 -*-

# File-Name:    fsSpikeTaskphaseDecodeIV_boot.py
# Version:      0.0.9 (see end of script for version descriptions)

# Import packages
import sys
import pandas as pd
pd.options.mode.chained_assignment = None # Turn off warnings for chained assignments
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import itertools
import matplotlib.pyplot as plt
import os
from statsmodels.stats.multitest import fdrcorrection
from itertools import combinations
import seaborn as sns
from itertools import product
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample
import argparse

# Import own packages
sys.path.append('...')
from Functions.Import.LocationsI import spikesDataII, resultsFolderI
from Functions.Preprocessing.dictionariesI import period_plot_name_dictI
from PreAnalysis.fsSpike_GallC_100ms_StimOn_Go_II import anchortimes, import_processed_spike_data, create_time_window_names
from Functions.Preprocessing.MazeMetadataI import mazes


# Locations
resultsLoc = resultsFolderI+"FullSetExplore/fsSpikeTaskphaseDecodeIV_boot/"
import_loc = spikesDataII+'230413/'

# Import precomputed data and windownames
analysisdf_import = import_processed_spike_data()
timewindow_names = create_time_window_names(anchortimes)

# Create dict with information of windows used
grouped_projection_windows = {
    'Baseline_Goal':('2_all',['TRGDLY_-0.2','TRGDLY_-0.1']),
    'Baseline_C1':('2_all',['UNFIXTG11_ON_-0.2','UNFIXTG11_ON_-0.1',]),
    'EarlyC1':('2_all',['UNFIXTG11_ON_-0.2','UNFIXTG11_ON_-0.1', 'UNFIXTG11_ON_0.0']),
    'LateC1':('2_all',['UNFIXTG11_ON_0.2', 'UNFIXTG11_ON_0.3']),
    'GoC1':('2_all',['FIXTG0_OFF_-0.2','FIXTG0_OFF_-0.1', 'FIXTG0_OFF_0.0']),
    'EarlyC2':('2_all',['UNFIXTG21_ON_-0.2', 'UNFIXTG21_ON_-0.1', 'UNFIXTG21_ON_0.0']),
    'LateC2':('2_all',['UNFIXTG21_ON_0.2', 'UNFIXTG21_ON_0.3']),
    'GoC2':('2_all',['FIXTG11_OFF_-0.2', 'FIXTG11_OFF_-0.1', 'FIXTG11_OFF_0.0']),
}

# Area data, updated without all as computation takes too long
'''
areas_sweep = [
    (['dPFC'],'dPFC'),(['dmPFC'],'dmPFC'),(['vPFC'],'vPFC'),(['Insula'],'Insula'),(['dlPM'],'dlPM'),
    (['PS'],'PS'),(['dPFC, dmPFC','dlPM','PS','Insula','vPFC'],'all')
]
'''
areas_sweep = [
    (['dPFC'],'dPFC'),(['dmPFC'],'dmPFC'),(['vPFC'],'vPFC'),(['Insula'],'Insula'),(['dlPM'],'dlPM'),
    (['PS'],'PS')]

# Add unit name information to analysisdf_import
analysisdf_import['Unit']=analysisdf_import['session']+'_'+analysisdf_import['NeuronNum'].astype(str)

# Add stratification variable to analysisdf_import, used for test/train split
goal_locations = [7,9,17,19]
analysisdf_import['stratificationVar'] = analysisdf_import['session']+'_'+analysisdf_import['NeuronNum'].astype(str)+'_'+analysisdf_import['FinalPos'].astype(str)

# Create list with all combinations of Goals and grouped_projection_windows.keys()
path_group_combinations = list(itertools.product(goal_locations, grouped_projection_windows.keys()))
path_group_combinations = [str(x[0])+'_'+x[1] for x in path_group_combinations]

# Filter out baseline keys from dictionary
non_baseline_keys = [key for key in grouped_projection_windows.keys() if not key.startswith('Baseline')]
# Create combinations using filtered keys
path_group_combinations_noBaseline = list(itertools.product(goal_locations, non_baseline_keys))
path_group_combinations_noBaseline = [str(x[0])+'_'+x[1] for x in path_group_combinations_noBaseline]

#### Functions
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-area_num", type=int, 
                        help="Area for analysis, chosen as numerical from list")
    args = parser.parse_args()
    area_num = args.area_num
    return area_num

def group_length(group):
    '''
    Function to count how many rows belong to a groupby group
    '''
    group['len']=len(group)
    return group

def create_cv_results_frame(analysisdf, resultsLoc, path_group_combinations, area_info, unit_list_input, grouped_projection_windows):
    area_list, area_name = area_info
    
    # Create train/test split
    analysisdf.reset_index(inplace=True, drop = True)
    df_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=50)
    X = list(range(len(analysisdf)))
    Y = analysisdf['stratificationVar']
    for i_split, (train_index, test_index) in enumerate(df_splitter.split(X,Y)):
        analysisdf.loc[train_index,'data'] = 'train'
        analysisdf.loc[test_index,'data'] = 'test'
        print('Split created')

    # Create results dataframe with path_group_combinations as index and columns, for cross correlation matrix
    cv_results_frame = pd.DataFrame(index=path_group_combinations, columns=path_group_combinations, dtype='float64')

    # Create means for each pathway and choice point
    data_dict = {}
    go_times_list = ['FIXTG0_OFF','FIXTG11_OFF','FIXTG21_OFF','FIXTG31_OFF']

    # Get multiindex to fill data
    unit_list = unit_list_input.copy()
    unit_list.sort()
    var_combinations = [7,9,17,19]
    var_combinations = list(product(unit_list,var_combinations, ['train','test']))
    var_combinations = [(i[0],i[1],i[2]) for i in var_combinations]
    multi_index = pd.MultiIndex.from_tuples(var_combinations, names=['Unit', 'FinalPos', 'data'])   

    ## First choice point
    grouping_data = analysisdf.copy()
    ### Exclude missing values and calculate means
    grouping_data = grouping_data.loc[(grouping_data['Choice2step']==1)]
    grouping_data = grouping_data.dropna(subset=[go_times_list[1]+'_-0.2'])
    grouping_data = grouping_data.groupby(['Unit','FinalPos','data']).mean()

    # Update index to full index
    grouped_data_full = grouping_data.reindex(multi_index)
    # Fill missing values with group mean per Unit
    means = grouped_data_full.groupby(['Unit']).transform('mean')
    grouped_data_full = grouped_data_full.fillna(means)
    grouping_data = grouped_data_full.reset_index()

    ### Split data again by train and test data
    train_df = grouping_data.loc[grouping_data['data']=='train']
    test_df = grouping_data.loc[grouping_data['data']=='test']
    ### Add data to dict
    data_dict['2_all']=(train_df,test_df)

    # Delete test_df and train_df so that variables can be used again later on savefly
    del train_df, test_df

    # Loop over all dfs in data_dict and create a list with units that are present in all dfs
    unit_list = analysisdf['Unit'].unique()

    # Get population normalisation vectors
    norm_data = analysisdf.copy()
    norm_vecs = {}
    for data_type in ['train','test']:
        norm_data_temp = norm_data.loc[norm_data['data']==data_type]
        norm_data_temp = norm_data_temp.groupby(['Unit']).mean().reset_index()
        unit_order = norm_data_temp['Unit']

        # Calculate norm vec
        norm_start_colum, norm_end_column = 'TRGDLY_-0.3', 'FIXTG11_OFF_0.0'
        normalisation_start_index = list(norm_data_temp.columns).index(norm_start_colum)
        normalisation_end_index = list(norm_data_temp.columns).index(norm_end_column)
        normalisation_length = normalisation_end_index-normalisation_start_index
        normalisation_term = norm_data_temp.iloc[:,normalisation_start_index:normalisation_end_index].sum(axis=1)
        normalisation_term = normalisation_term/normalisation_length
        norm_vecs[data_type] = (unit_order, normalisation_term)

    # Dicts to hold population activity vectors for both train and test data
    pop_vec_dict = {
        'train':{},
        'test':{}
    }

    # Loop over path_group_combinations to fill both train_pop_vec_dict and test_pop_vec_dict
    for curr_path_group in path_group_combinations:

        # Get current goal and window
        curr_goal = curr_path_group.split('_',1)[0]
        curr_window = curr_path_group.split('_',1)[-1]

        # Select correct type of data (by step group) depending on window
        curr_data_name, curr_wins2mean = grouped_projection_windows[curr_window]
        curr_data_tuple = data_dict[curr_data_name]

        # Loop over test and train
        for i_curr_data_type, curr_data_type in enumerate(['train','test']):
            curr_data = curr_data_tuple[i_curr_data_type]

            # Calculate mean window activity for each unit
            curr_data['mean_window'] = curr_data[curr_wins2mean].mean(axis=1)

            # Select only units that are present in all dfs
            curr_data = curr_data.loc[curr_data['Unit'].isin(unit_list)]

            # Assert the unit order is the same as in norm_vecs
            assert curr_data.loc[curr_data['FinalPos']==int(curr_goal), 'Unit'].values.all() == norm_vecs[curr_data_type][0].all()

            # Select relevant population activity vector
            curr_pop_vec = curr_data.loc[curr_data['FinalPos']==int(curr_goal), 'mean_window'].values

            # Normalise population activity vector
            curr_pop_vec = curr_pop_vec/norm_vecs[curr_data_type][1]

            # Add population activity vector to pop_vec_dict
            pop_vec_dict[curr_data_type][curr_path_group] = curr_pop_vec

    # Correlate each vector in pop_vec_dict['train'] with each vector in pop_vec_dict['test'] and save in cv_results_frame
    for train_pop_vec_name, test_pop_vec_name in itertools.product(pop_vec_dict['train'].keys(), pop_vec_dict['test'].keys()):
        train_pop_vec = pop_vec_dict['train'][train_pop_vec_name]
        test_pop_vec = pop_vec_dict['test'][test_pop_vec_name]
        # Remove faulty values from both vectors, reulting from normalisation (division by zero)
        ### Delete nan from train in both
        test_pop_vec = test_pop_vec[~np.isnan(train_pop_vec)]
        train_pop_vec = train_pop_vec[~np.isnan(train_pop_vec)]
        ### Delete nan from test in both
        train_pop_vec = train_pop_vec[~np.isnan(test_pop_vec)]
        test_pop_vec = test_pop_vec[~np.isnan(test_pop_vec)]

        # Calculate correlation
        correlation = np.corrcoef(train_pop_vec, test_pop_vec)[0,1]
        # Assert that correlation is not nan
        assert not np.isnan(correlation)
        cv_results_frame.loc[train_pop_vec_name, test_pop_vec_name] = correlation

    cv_results_frame.to_csv(resultsLoc+area_name+'_cv_results_frame.csv')
    return cv_results_frame


def png_plot(cv_results_frame, path_group_combinations, goal_locations, area_name):
    # Create plot
    plt.figure(figsize=(10,10))

    # Plot correlation matrix
    corr_matrix = cv_results_frame.values
    plt.imshow(corr_matrix)
    plt.colorbar()
    plt.clim(-0.6,0.6)
    # Change color scheme to PRGn
    plt.set_cmap('PRGn')



    # Extract anchorlines indices
    ## Pathline for every time the pathgroup in path_group_combinations changes
    pathline_indices = [x.rsplit('_',2)[0] for x in path_group_combinations]
    pathline_indices = [(str(x),pathline_indices.index(str(x))-0.5) for x in goal_locations]
    ## Small pathline for every time it is an early time period
    small_pathline_indices = [x.split('_',1)[1] for x in path_group_combinations]
    small_pathline_indices_print = []
    for i_num, i_item in enumerate(small_pathline_indices):
        if i_item.startswith('Early'):
            small_pathline_indices_print.append((i_item[5:],i_num-0.5))

    # Add anchorlines to plot
    ## Pathline_indices
    axis_plot_names = []
    axis_plot_location = []
    for i_anchortime, i_anchortime_vars in pathline_indices:
        plt.axvline(x=i_anchortime_vars,color='black', linewidth = 3)
        plt.axhline(y=i_anchortime_vars,color='black', linewidth = 3)
        axis_plot_names.append(i_anchortime)
        axis_plot_location.append(i_anchortime_vars)
    ## Small_pathline_indices
    anchortimes_plot_names = []
    anchortimes_plot_ticks = []
    for i_anchortime, i_anchortime_vars in small_pathline_indices_print:
        plt.axvline(x=i_anchortime_vars,color='black', linestyle=':', linewidth = 2)
        plt.axhline(y=i_anchortime_vars,color='black', linestyle=':', linewidth = 2)
        anchortimes_plot_names.append(i_anchortime)
        anchortimes_plot_ticks.append(i_anchortime_vars)

    # Set ticks to match lines
    plt.xticks(anchortimes_plot_ticks, anchortimes_plot_names, rotation=-90)
    plt.yticks(anchortimes_plot_ticks, anchortimes_plot_names)

    # Get height of plot
    plot_height = corr_matrix.shape[0]

    # Add the axis_plot_names outside of the plot at corresponding locations, on the left and bottom
    for i_anchortime, i_anchortime_vars in zip(axis_plot_names, axis_plot_location):
        plt.text(i_anchortime_vars+7.5, 80, i_anchortime, color='red', fontsize=15)
        plt.text(-12.5, i_anchortime_vars+8.5, i_anchortime, color='red', fontsize=15, rotation=90)

    # Add title
    plt.title('Taskphase controlled, by goal, area: '+area_name, size=15)

    plt.savefig(resultsLoc+'plot_{0}.png'.format(area_name), dpi=300)
    plt.close()
    
def svg_plot(cv_results_frame, path_group_combinations, goal_locations, area_name):
    # Create plot
    plt.figure(figsize=(2.2,2.2))

    SMALL_SIZE = 6
    MEDIUM_SIZE = 6
    BIGGER_SIZE = 6

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Plot correlation matrix
    corr_matrix = cv_results_frame.values
    plt.imshow(corr_matrix)
    plt.colorbar()
    plt.clim(-0.6,0.6)

    ticks_dict = {
        'Goal':'Goal',
        'Earl':'C1',
        'C2_2step':'C2 (goal)',
        'C2_4step':'C2 (away)',
        'C3_4step':'C3',
        'C4_4step':'C4 (goal)'
    }

    # Extract anchorlines indices
    ## Pathline for every time the pathgroup in path_group_combinations changes
    pathline_indices = [x.rsplit('_',2)[0] for x in path_group_combinations]
    pathline_indices = [(str(x),pathline_indices.index(str(x))-0.5) for x in goal_locations]
    ## Small pathline for every time it is an early time period
    small_pathline_indices = [x.split('_',1)[1] for x in path_group_combinations]
    small_pathline_indices_print = []
    for i_num, i_item in enumerate(small_pathline_indices):
        if i_item.startswith('Early'):
            small_pathline_indices_print.append((i_item[5:],i_num-0.5))

    # Add anchorlines to plot
    ## Pathline_indices
    axis_plot_names = []
    axis_plot_location = []
    for i_anchortime, i_anchortime_vars in pathline_indices:
        plt.axvline(x=i_anchortime_vars,color='black', linewidth = 1.5)
        plt.axhline(y=i_anchortime_vars,color='black', linewidth = 1.5)
        axis_plot_names.append(i_anchortime)
        axis_plot_location.append(i_anchortime_vars)
    ## Small_pathline_indices
    anchortimes_plot_names = []
    anchortimes_plot_ticks = []
    for i_anchortime, i_anchortime_vars in small_pathline_indices_print:
        plt.axvline(x=i_anchortime_vars,color='black', linestyle=':', linewidth = 0.75)
        plt.axhline(y=i_anchortime_vars,color='black', linestyle=':', linewidth = 0.75)
        anchortimes_plot_names.append(i_anchortime)
        anchortimes_plot_ticks.append(i_anchortime_vars)

    # Set ticks to match lines
    plt.xticks(anchortimes_plot_ticks, anchortimes_plot_names, rotation=-90)
    plt.yticks(anchortimes_plot_ticks, anchortimes_plot_names)

    # Get height of plot
    plot_height = corr_matrix.shape[0]

    # Add the axis_plot_names outside of the plot at corresponding locations, on the left and bottom
    #for i_anchortime, i_anchortime_vars in zip(axis_plot_names, axis_plot_location):
    #    plt.text(i_anchortime_vars+7.5, 80, i_anchortime, color='red', fontsize=15)
    #    plt.text(-12.5, i_anchortime_vars+8.5, i_anchortime, color='red', fontsize=15, rotation=90)

    # Add title
    plt.title('Taskphase controlled, by goal, area: '+area_name)

    #Make font editable in svg
    plt.rcParams['svg.fonttype'] = 'none'
    # Set font to arial
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'arial'

    plt.savefig(resultsLoc+'print/'+'plot_{0}.svg'.format(area_name), format='svg',dpi=300, bbox_inches="tight")
    plt.close()


def create_thesis_summary_metrics(cv_results_frame, goal_locations, area_name, create_figures = False):
    combined_summary_metrics = {}
    def pairings_loop(cv_results_frame, goal_pairings, choice_pairings):
        val_list = []
        for i_goal_pairing, i_choice_pairing in itertools.product(goal_pairings, choice_pairings):
            curr_index = str(i_goal_pairing[0])+'_'+i_choice_pairing[0]
            curr_column = str(i_goal_pairing[1])+'_'+i_choice_pairing[1]
            curr_val = cv_results_frame.loc[curr_index,curr_column]
            val_list.append(curr_val)
        return val_list

    ######## Create within and across goal combinations
    within_goals = list(zip(goal_locations,goal_locations))
    across_goals_order = list(combinations(goal_locations,2))
    across_goals_reversed = [(x[1],x[0]) for x in across_goals_order]
    across_goals = across_goals_order + across_goals_reversed
    list_of_goal_pairings = [('within',within_goals),('across',across_goals)]

    for goal_pairings_name, goal_pairings in list_of_goal_pairings:
        summary_metrics = {}
        #### Same choice and same window
        relevant_windows = ['EarlyC1','LateC1','GoC1','EarlyC2','LateC2','GoC2']
        choice_pairings = list(zip(relevant_windows,relevant_windows))
        val_list = pairings_loop(cv_results_frame, goal_pairings, choice_pairings)

        summary_metrics['sCsW'] = np.tanh(np.mean(np.arctanh(val_list)))

        #### Different choice and same window
        relevant_windows = [['EarlyC1','LateC1','GoC1'],['EarlyC2','LateC2','GoC2']]
        choice_pairings = list(zip(relevant_windows[0]+relevant_windows[1],relevant_windows[1]+relevant_windows[0]))
        val_list = pairings_loop(cv_results_frame, goal_pairings, choice_pairings)
        summary_metrics['dCsW'] = np.tanh(np.mean(np.arctanh(val_list)))

        #### Same choice and different window
        other_window_combinations = list(combinations(['Early','Late','Go'],2))
        other_window_combinations_C1 = [(x[0]+'C1',x[1]+'C1') for x in other_window_combinations]
        other_window_combinations_C2 = [(x[0]+'C2',x[1]+'C2') for x in other_window_combinations]
        other_window_combinations = other_window_combinations_C1 + other_window_combinations_C2
        reversed_other_window_combinations = [(x[1],x[0]) for x in other_window_combinations]
        choice_pairings = other_window_combinations + reversed_other_window_combinations
        val_list = pairings_loop(cv_results_frame, goal_pairings, choice_pairings)
        summary_metrics['sCdW'] = np.tanh(np.mean(np.arctanh(val_list)))

        #### Different choice and different window
        other_window_combinations = list(combinations(['Early','Late','Go'],2))
        other_window_combinations_C1 = [(x[0]+'C1',x[1]+'C2') for x in other_window_combinations]
        other_window_combinations_C2 = [(x[0]+'C2',x[1]+'C1') for x in other_window_combinations]
        other_window_combinations = other_window_combinations_C1 + other_window_combinations_C2
        reversed_other_window_combinations = [(x[1],x[0]) for x in other_window_combinations]
        choice_pairings = other_window_combinations + reversed_other_window_combinations
        val_list = pairings_loop(cv_results_frame, goal_pairings, choice_pairings)
        summary_metrics['dCdW'] = np.tanh(np.mean(np.arctanh(val_list)))

        # Add summary_metrics to combined_summary_metrics
        combined_summary_metrics[goal_pairings_name] = summary_metrics

    # Create pandas df from combined_summary_metrics
    combined_summary_metrics_df = pd.DataFrame.from_dict(combined_summary_metrics)
    # Unstack combined_summary_metrics_df
    combined_summary_metrics_df = combined_summary_metrics_df.unstack().reset_index()
    # Rename columns
    combined_summary_metrics_df.columns = ['grouping','metric','value']

    if create_figures:
        # Make figure with barplots split by Choice
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
        spine_linewidth = 0.65


        # Use two colors from viridis, purple and green
        color1 = (33/255, 145/255, 140/255)  # RGB values normalized
        color2 = (68/255, 1/255, 84/255)     # RGB values normalized

        # Create a list of colors for the palette
        custom_palette = [color1, color2]

        # Set the color palette in Seaborn
        sns.set_palette(custom_palette)

        # Create bar chart from combined_summary_metrics_df usinfg seaborn
        plt.figure(figsize=(1.1,1.8))
        sns.barplot(data=combined_summary_metrics_df, y='metric', x='value', hue='grouping')
        plt.xlabel('Average correlation')
        plt.ylabel(' ')
        plt.title('Area: '+area_name)
        plt.legend(edgecolor='dimgray')

        # Only have axis on x = 0, no other spines
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        #ax.spines['left'].set_position(('data', 0))

        # Add vertical lines to indicate 0
        plt.axvline(x=0, color='black', linewidth=0.75)

        # Make bottom spine to spine_linewidth
        ax.spines['bottom'].set_linewidth(spine_linewidth)

        # Keep y ticks labels but remove ticks
        plt.tick_params(axis='y', length=0)

        # Make x ticks match spine_linewidth
        ax.tick_params(axis='x', width=spine_linewidth)

        # Set x axis limits to -0.2 to 0.65
        plt.xlim(-0.2,0.65)

        # Set x axis 0 tick to
        plt.xticks([0,0.25,0.5])
        
        #Make font editable in svg
        plt.rcParams['svg.fonttype'] = 'none'
        # Set font to arial
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'arial'

        # Save figure
        plt.savefig(resultsLoc+'print/metricbars2_{0}.svg'.format(area_name), format='svg',dpi=300, bbox_inches="tight")

        # Save figure
        plt.tight_layout()
        plt.savefig(resultsLoc+'metricbars2_{0}.png'.format(area_name),dpi=300, bbox_inches="tight")
        plt.close()
    return combined_summary_metrics_df 

def calculate_contrast_metrics(summary_metrics_df):
    '''
    1. within - across

    (for the following, use average of within and across)
    2. sCsW - dCdW
    3. dCsW - dCdW
    4. sCdW - dCdW
    5. dCsW - sCdW
    '''
    contrast_metrics_dict = {}
    # Within - across
    within_metrics = summary_metrics_df.loc[summary_metrics_df['grouping']=='within','value']
    within_metrics = np.mean(np.arctanh(within_metrics.values))
    across_metrics = summary_metrics_df.loc[summary_metrics_df['grouping']=='across','value']
    across_metrics = np.mean(np.arctanh(across_metrics.values))
    contrast_metrics_dict['within-across'] = np.tanh(within_metrics-across_metrics)

    # sCsW - dCdW
    sCsW = summary_metrics_df.loc[(summary_metrics_df['metric']=='sCsW'),'value']
    sCsW = np.mean(np.arctanh(sCsW.values))
    dCdW = summary_metrics_df.loc[(summary_metrics_df['metric']=='dCdW'),'value']
    dCdW = np.mean(np.arctanh(dCdW.values))
    contrast_metrics_dict['sCsW-dCdW'] = np.tanh(sCsW-dCdW)

    # dCsW - dCdW
    dCsW = summary_metrics_df.loc[(summary_metrics_df['metric']=='dCsW'),'value']
    dCsW = np.mean(np.arctanh(dCsW.values))
    contrast_metrics_dict['dCsW-dCdW'] = np.tanh(dCsW-dCdW)

    # sCdW - dCdW
    sCdW = summary_metrics_df.loc[(summary_metrics_df['metric']=='sCdW'),'value']
    sCdW = np.mean(np.arctanh(sCdW.values))
    contrast_metrics_dict['sCdW-dCdW'] = np.tanh(sCdW-dCdW)

    # dCsW - sCdW
    contrast_metrics_dict['dCsW-sCdW'] = np.tanh(dCsW-sCdW)

    return contrast_metrics_dict

#### Execution
print('fsSpikeTaskphaseDecodeIV_boot.py')
area_num = parse_args()
area_info = areas_sweep[area_num]
#area_info = areas_sweep[2]
area_list = area_info[0]
analysisdf = analysisdf_import.loc[analysisdf_import['area'].isin(area_list)].reset_index(drop=True)
cv_results_frame = create_cv_results_frame(analysisdf, resultsLoc, path_group_combinations, area_info, analysisdf['Unit'].unique(), grouped_projection_windows)
# Filter out columns & rows that don't contain 'Baseline'; these should not influence the plot
filtered_df = cv_results_frame.loc[:, ~cv_results_frame.columns.str.contains('Baseline')]
filtered_df = filtered_df.loc[~filtered_df.index.str.contains('Baseline')]
cv_results_frame_backup = cv_results_frame.copy()
cv_results_frame = filtered_df
png_plot(cv_results_frame, path_group_combinations_noBaseline, goal_locations, area_info[1])
svg_plot(cv_results_frame, path_group_combinations_noBaseline, goal_locations, area_info[1])
summary_metrics_df = create_thesis_summary_metrics(cv_results_frame, goal_locations, area_info[1], create_figures = True)
contrast_metrics_dict = calculate_contrast_metrics(summary_metrics_df)

# Create df based on contrast_metrics_dict, with additional column for p-values and corrected p-values
contrast_metrics_df = pd.DataFrame.from_dict(contrast_metrics_dict, orient='index', columns=['value'])
contrast_metrics_df['p_value'] = np.nan
contrast_metrics_df['corrected_p_value'] = np.nan

# Create dict with same keys but empty lists as values
contrast_dict_bootstrap = {}
for i_key in contrast_metrics_dict.keys():
    contrast_dict_bootstrap[i_key] = []

# Get John's contrast value, within goal location
confounding_controls = {
    'Goal_C1':[('Baseline_Goal','Baseline_C1'),('Baseline_C1','Baseline_Goal')],
    'EarlyC1_LateC1':[('EarlyC1','LateC1'),('LateC1','EarlyC1')],
    'LateC1_GoC1':[('LateC1','GoC1'),('GoC1','LateC1')],
    'EarlyC1_GoC1':[('EarlyC1','GoC1'),('GoC1','EarlyC1')]
}
confounding_controls_list = {
    'Goal_C1':[],
    'EarlyC1_LateC1':[],
    'LateC1_GoC1':[],
    'EarlyC1_GoC1':[]
}
confounding_controls_contrasts = {
    'Goal_C1-EarlyC1_LateC1':[],
    'Goal_C1-LateC1_GoC1':[],
    'Goal_C1-EarlyC1_GoC1':[]
}
confounding_controls_contrasts_refValues = {
    'Goal_C1-EarlyC1_LateC1':None,
    'Goal_C1-LateC1_GoC1':None,
    'Goal_C1-EarlyC1_GoC1':None
}

for i_control in confounding_controls.keys():
    for i_goal in goal_locations:
        for i_pair in confounding_controls[i_control]:
            confounding_controls_list[i_control].append(
                cv_results_frame_backup.loc['{}_{}'.format(i_goal,i_pair[0]),'{}_{}'.format(i_goal,i_pair[1])]
            )
# Old uncorrected version
#for i_contrast in confounding_controls_contrasts.keys():
#    contrast_part1, contrast_part2 = i_contrast.split('-')
#    confounding_controls_contrasts_refValues[i_contrast] = np.mean(confounding_controls_list[contrast_part1])-np.mean(confounding_controls_list[contrast_part2])
# New corrected version
for i_contrast in confounding_controls_contrasts.keys():
    contrast_part1, contrast_part2 = i_contrast.split('-')
    confounding_controls_contrasts_refValues[i_contrast] = np.tanh(
        np.mean(np.arctanh(confounding_controls_list[contrast_part1])) - 
        np.mean(np.arctanh(confounding_controls_list[contrast_part2]))
    )

# Save df with mean of each list to disk
confounding_controls_export = pd.DataFrame(index=list(confounding_controls_list.keys()))
for i_contrast in confounding_controls_list.keys():
    confounding_controls_export.loc[i_contrast,'value'] = np.tanh(np.mean(np.arctanh(confounding_controls_list[i_contrast])))
confounding_controls_export.to_csv(resultsLoc+area_info[1]+'_baseline_values.csv')

# Created bootstrapped projections
analysisdf.reset_index(inplace=True)
#stratification_var = analysisdf['Unit'].astype(str)

for i_split in range(1000):
    # Resample data
    #sampled_index = resample(analysisdf.index, replace=True, n_samples=len(analysisdf), stratify=stratification_var)
    print(f"Fold {i_split}:")
    #analysisdf_boot = analysisdf.copy().iloc[sampled_index]
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

    # Run analysis on bootstrapped data
    cv_results_frame_boot = create_cv_results_frame(analysisdf_boot, resultsLoc, path_group_combinations, area_info, unit_list, grouped_projection_windows)
    # Filter out columns & rows that don't contain 'Baseline'; these should not influence analysis
    filtered_df = cv_results_frame_boot.loc[:, ~cv_results_frame_boot.columns.str.contains('Baseline')]
    filtered_df = filtered_df.loc[~filtered_df.index.str.contains('Baseline')]
    cv_results_frame_boot_backup = cv_results_frame_boot.copy()
    cv_results_frame_boot = filtered_df

    summary_metrics_df_boot = create_thesis_summary_metrics(cv_results_frame_boot, goal_locations, area_info[1], create_figures = False)
    contrast_metrics_dict_boot = calculate_contrast_metrics(summary_metrics_df_boot)

    ## Add each list from results_dict to boot_results_dict
    for key, value in contrast_metrics_dict_boot.items():
        contrast_dict_bootstrap[key].append(value)

    # Get John's contrast values
    confounding_controls_list = {
        'Goal_C1':[],
        'EarlyC1_LateC1':[],
        'LateC1_GoC1':[],
        'EarlyC1_GoC1':[]
    }
    for i_control in confounding_controls.keys():
        for i_goal in goal_locations:
            for i_pair in confounding_controls[i_control]:
                confounding_controls_list[i_control].append(
                    cv_results_frame_boot_backup.loc['{}_{}'.format(i_goal,i_pair[0]),'{}_{}'.format(i_goal,i_pair[1])]
                )
    for i_contrast in confounding_controls_contrasts.keys():
        contrast_part1, contrast_part2 = i_contrast.split('-')
        # Old uncorrected version
        #confounding_controls_contrasts[i_contrast].append(
        #    np.mean(confounding_controls_list[contrast_part1])-np.mean(confounding_controls_list[contrast_part2])
        #)
        # New corrected version
        confounding_controls_contrasts[i_contrast].append(
            np.tanh(np.mean(np.arctanh(confounding_controls_list[contrast_part1])) - 
                   np.mean(np.arctanh(confounding_controls_list[contrast_part2])))
        )


# Add Johns contrast value to dicts for them to be included in p value calculation
for i_contrast in confounding_controls_contrasts.keys():
    contrast_metrics_dict[i_contrast] = confounding_controls_contrasts_refValues[i_contrast]
    contrast_dict_bootstrap[i_contrast] = confounding_controls_contrasts[i_contrast]
    contrast_metrics_df.loc[i_contrast,'value'] = confounding_controls_contrasts_refValues[i_contrast]


# Extract p_values, old routine
'''
for i_comparison in contrast_metrics_dict.keys():
    baseline_val = contrast_metrics_dict[i_comparison]
    bootstrapped_vals = np.array(contrast_dict_bootstrap[i_comparison])
    if baseline_val>0:
        p_value = np.sum(bootstrapped_vals<0)/len(bootstrapped_vals)
    elif baseline_val<0:
        p_value = np.sum(bootstrapped_vals>0)/len(bootstrapped_vals)
    else:
        raise ValueError('Baseline value is 0')
    if p_value>0.5:
        p_value = 1-p_value
    p_value = p_value*2
    contrast_metrics_df.loc[i_comparison,'p_value'] = p_value
'''
for i_comparison in contrast_metrics_dict.keys():
    baseline_val = contrast_metrics_dict[i_comparison]
    bootstrapped_vals = np.array(contrast_dict_bootstrap[i_comparison])
    # When baseline is 0, there should be no significant difference
    if baseline_val == 0:
        p_value = 1.0  # Not significant
    else:
        # Calculate p-value based on direction of effect
        if baseline_val > 0:
            p_value = np.sum(bootstrapped_vals<0)/len(bootstrapped_vals)
        else:  # baseline_val < 0
            p_value = np.sum(bootstrapped_vals>0)/len(bootstrapped_vals)
        
        # Convert to two-tailed p-value
        if p_value > 0.5:
            p_value = 1 - p_value
        p_value = p_value * 2
        contrast_metrics_df.loc[i_comparison,'p_value'] = p_value

# Correct p_values, alpha = 0.05
corrected_p_values = multipletests(contrast_metrics_df['p_value'], alpha = 0.05, method='fdr_bh')[1]
contrast_metrics_df['corrected_p_value_0-05'] = corrected_p_values
# Correct p_values, alpha = 0.01
corrected_p_values = multipletests(contrast_metrics_df['p_value'], alpha = 0.01, method='fdr_bh')[1]
contrast_metrics_df['corrected_p_value_0-01'] = corrected_p_values

# Save contrast_metrics_df
contrast_metrics_df.to_csv(resultsLoc+area_info[1]+'_contrast_metrics.csv')
