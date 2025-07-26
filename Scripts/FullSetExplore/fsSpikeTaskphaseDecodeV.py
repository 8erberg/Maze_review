# -*- coding: utf-8 -*-

# File-Name:    fsSpikeTaskphaseDecodeV.py
# Version:      0.0.3 (see end of script for version descriptions)

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
resultsLoc = resultsFolderI+"FullSetExplore/fsSpikeTaskphaseDecodeV/"
import_loc = spikesDataII+'230413/'

# Import precomputed data and windownames
analysisdf_import = import_processed_spike_data()
timewindow_names = create_time_window_names(anchortimes)

# Create dict with information of windows used
grouped_projection_windows = {
    'EarlyC1':('2_all',['UNFIXTG11_ON_-0.2','UNFIXTG11_ON_-0.1', 'UNFIXTG11_ON_0.0']),
    'LateC1':('2_all',['UNFIXTG11_ON_0.2', 'UNFIXTG11_ON_0.3']),
    'GoC1':('2_all',['FIXTG0_OFF_-0.2','FIXTG0_OFF_-0.1', 'FIXTG0_OFF_0.0']),
    'EarlyC2':('2_all',['UNFIXTG21_ON_-0.2', 'UNFIXTG21_ON_-0.1', 'UNFIXTG21_ON_0.0']),
    'LateC2':('2_all',['UNFIXTG21_ON_0.2', 'UNFIXTG21_ON_0.3']),
    'GoC2':('2_all',['FIXTG11_OFF_-0.2', 'FIXTG11_OFF_-0.1', 'FIXTG11_OFF_0.0']),
}

areas_sweep = [
    (['dPFC'],'dPFC'),(['dmPFC'],'dmPFC'),(['vPFC'],'vPFC'),(['Insula'],'Insula'),(['dlPM'],'dlPM'),
    (['PS'],'PS')]

# Add unit name information to analysisdf_import
analysisdf_import['Unit']=analysisdf_import['session']+'_'+analysisdf_import['NeuronNum'].astype(str)

# Create list with all Pathways and grouped_projection_windows.keys()
pathway_numbers = mazes.loc[mazes['Nsteps']==2].Npath.unique()
path_group_combinations = list(itertools.product(pathway_numbers, grouped_projection_windows.keys()))
path_group_combinations = [str(x[0])+'_'+x[1] for x in path_group_combinations]

# Reduce data to 2 step problems
analysisdf_import = analysisdf_import.loc[analysisdf_import['PathNum'].isin(pathway_numbers)]

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

    # Create results dataframe with path_group_combinations as index and columns, for cross correlation matrix
    cv_results_frame = pd.DataFrame(index=path_group_combinations, columns=path_group_combinations, dtype='float64')

    # Get multiindex to fill data
    unit_list = unit_list_input.copy()
    unit_list.sort()
    var_combinations = list(product(unit_list,pathway_numbers))
    var_combinations = [(i[0],i[1]) for i in var_combinations]
    multi_index = pd.MultiIndex.from_tuples(var_combinations, names=['Unit', 'PathNum'])   

    ### Exclude missing values and calculate means
    grouping_data = analysisdf.copy()
    grouping_data = grouping_data.loc[(grouping_data['Choice2step']==1)]
    grouping_data = grouping_data.dropna(subset=['FIXTG11_OFF'+'_-0.2'])
    grouping_data = grouping_data.groupby(['Unit','PathNum']).mean()

    # Update index to full index
    grouped_data_full = grouping_data.reindex(multi_index)
    # Fill missing values with group mean per Unit
    means = grouped_data_full.groupby(['Unit']).transform('mean')
    grouped_data_full = grouped_data_full.fillna(means)
    grouping_data = grouped_data_full.reset_index()
    corr_df = grouping_data.copy()

    # Get population normalisation vectors
    norm_data_temp = analysisdf.copy()
    norm_data_temp = norm_data_temp.groupby(['Unit']).mean().reset_index()
    unit_order = norm_data_temp['Unit']

    # Calculate norm vec
    norm_start_colum, norm_end_column = 'TRGDLY_-0.3', 'FIXTG11_OFF_0.0'
    normalisation_start_index = list(norm_data_temp.columns).index(norm_start_colum)
    normalisation_end_index = list(norm_data_temp.columns).index(norm_end_column)
    normalisation_length = normalisation_end_index-normalisation_start_index
    normalisation_term = norm_data_temp.iloc[:,normalisation_start_index:normalisation_end_index].sum(axis=1)
    normalisation_term = normalisation_term/normalisation_length
    norm_vec = (unit_order, normalisation_term)

    # Dicts to hold population activity vectors for both train and test data
    pop_vec_dict = {}

    # Loop over path_group_combinations to fill both train_pop_vec_dict and test_pop_vec_dict
    for curr_path_group in path_group_combinations:

        # Get current goal and window
        curr_pathwayNum = curr_path_group.split('_',1)[0]
        curr_window = curr_path_group.split('_',1)[-1]

        # Select correct type of data (by step group) depending on window
        _, curr_wins2mean = grouped_projection_windows[curr_window]
        curr_data = corr_df.copy()

        # Calculate mean window activity for each unit
        curr_data['mean_window'] = curr_data[curr_wins2mean].mean(axis=1)

        # Assert the unit order is the same as in norm_vecs
        assert curr_data.loc[curr_data['PathNum']==int(curr_pathwayNum), 'Unit'].values.all() == norm_vec[0].all()

        # Select relevant population activity vector
        curr_pop_vec = curr_data.loc[curr_data['PathNum']==int(curr_pathwayNum), 'mean_window'].values

        # Normalise population activity vector
        curr_pop_vec = curr_pop_vec/norm_vec[1]

        # Add population activity vector to pop_vec_dict
        pop_vec_dict[curr_path_group] = curr_pop_vec

    # Correlate each vector in pop_vec_dict['train'] with each vector in pop_vec_dict['test'] and save in cv_results_frame
    for train_pop_vec_name, test_pop_vec_name in itertools.product(pop_vec_dict.keys(), pop_vec_dict.keys()):
        train_pop_vec = pop_vec_dict[train_pop_vec_name]
        test_pop_vec = pop_vec_dict[test_pop_vec_name]
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


def create_summary_metrics(cv_results_frame_reduced, area_name, create_figures = False):
    def pairings_loop(cv_results_frame_reduced, choice_pairings):
        val_list = []
        for i_choice_pairing in choice_pairings:
            curr_index = i_choice_pairing[0]
            curr_column = i_choice_pairing[1]
            try:
                curr_val = cv_results_frame_reduced.loc[curr_index,curr_column]
                if curr_val == np.nan:
                    pass
                else:
                    val_list.append(curr_val)
            except KeyError:
                pass
        return val_list

    # Set double values to np.nan to avoid influence on means
    #cv_results_frame_reduced.loc['LateC2','EarlyC2'] = np.nan
    #cv_results_frame_reduced.loc['GoC2','EarlyC2'] = np.nan
    #cv_results_frame_reduced.loc['GoC2','LateC2'] = np.nan

    ######## Create metrics
    summary_metrics = {}
    #### Same choice and same window
    relevant_windows = ['EarlyC1','LateC1','GoC1','EarlyC2','LateC2','GoC2']
    choice_pairings = list(zip(relevant_windows,relevant_windows))
    val_list = pairings_loop(cv_results_frame_reduced, choice_pairings)
    summary_metrics['sCsW'] = np.tanh(np.mean(np.arctanh(val_list)))

    #### Different choice and same window
    relevant_windows = [['EarlyC1','LateC1','GoC1'],['EarlyC2','LateC2','GoC2']]
    choice_pairings = list(zip(relevant_windows[0]+relevant_windows[1],relevant_windows[1]+relevant_windows[0]))
    val_list = pairings_loop(cv_results_frame_reduced, choice_pairings)
    summary_metrics['dCsW'] = np.tanh(np.mean(np.arctanh(val_list)))

    #### Same choice and different window
    other_window_combinations = list(combinations(['Early','Late','Go'],2))
    other_window_combinations_C1 = [(x[0]+'C1',x[1]+'C1') for x in other_window_combinations]
    other_window_combinations_C2 = [(x[0]+'C2',x[1]+'C2') for x in other_window_combinations]
    other_window_combinations = other_window_combinations_C1 + other_window_combinations_C2
    reversed_other_window_combinations = [(x[1],x[0]) for x in other_window_combinations]
    choice_pairings = other_window_combinations + reversed_other_window_combinations
    val_list = pairings_loop(cv_results_frame_reduced, choice_pairings)
    summary_metrics['sCdW'] = np.tanh(np.mean(np.arctanh(val_list)))

    #### Different choice and different window
    other_window_combinations = list(combinations(['Early','Late','Go'],2))
    other_window_combinations_C1 = [(x[0]+'C1',x[1]+'C2') for x in other_window_combinations]
    other_window_combinations_C2 = [(x[0]+'C2',x[1]+'C1') for x in other_window_combinations]
    other_window_combinations = other_window_combinations_C1 + other_window_combinations_C2
    reversed_other_window_combinations = [(x[1],x[0]) for x in other_window_combinations]
    choice_pairings = other_window_combinations + reversed_other_window_combinations
    val_list = pairings_loop(cv_results_frame_reduced, choice_pairings)
    summary_metrics['dCdW'] = np.tanh(np.mean(np.arctanh(val_list)))

    # Create pandas df from summary_metrics - simpler now since no grouping
    summary_df = pd.DataFrame.from_dict(summary_metrics, orient='index', columns=['value'])
    summary_df = summary_df.reset_index()
    summary_df.columns = ['metric', 'value']

    if create_figures:
        # Plotting code
        SMALL_SIZE = 6
        MEDIUM_SIZE = 6
        BIGGER_SIZE = 7

        plt.rc('font', size=SMALL_SIZE)          
        plt.rc('axes', titlesize=SMALL_SIZE)     
        plt.rc('axes', labelsize=MEDIUM_SIZE)    
        plt.rc('xtick', labelsize=SMALL_SIZE)    
        plt.rc('ytick', labelsize=SMALL_SIZE)    
        plt.rc('legend', fontsize=SMALL_SIZE)    
        plt.rc('figure', titlesize=BIGGER_SIZE)  
        spine_linewidth = 0.65

        # Use two colors from viridis
        color1 = (33/255, 145/255, 140/255)  

        # Create the bar chart
        plt.figure(figsize=(1.6,1.2))
        sns.barplot(data=summary_df, y='metric', x='value', color=color1)
        plt.xlabel('Average correlation')
        plt.ylabel(' ')
        plt.title('Area: '+area_name)

        # Style the axes
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add vertical line at x=0
        plt.axvline(x=0, color='black', linewidth=0.75)

        # Style the spines and ticks
        ax.spines['bottom'].set_linewidth(spine_linewidth)
        plt.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', width=spine_linewidth)

        # Set axis limits and ticks
        plt.xlim(-0.2, 0.4)
        plt.xticks([-0.15,0, 0.15, 0.3])

        # Font settings
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'arial'

        # Save figures
        plt.savefig(resultsLoc+'print/metricbars2_{0}.svg'.format(area_name), 
                    format='svg', dpi=300, bbox_inches="tight")
        plt.tight_layout()
        plt.savefig(resultsLoc+'metricbars2_{0}.png'.format(area_name),
                    dpi=300, bbox_inches="tight")
        plt.close()
    return summary_df 

def calculate_contrast_metrics(summary_metrics_df):
    '''
    Calculate contrasts between metrics:
    1. sCsW - dCdW
    2. dCsW - dCdW
    3. sCdW - dCdW
    4. dCsW - sCdW
    5. sCsW - dCsW
    6. sCsW - sCdW
    '''
    contrast_metrics_dict = {}
    
    # Get values and apply Fisher z-transform
    sCsW = np.arctanh(summary_metrics_df.loc[summary_metrics_df['metric']=='sCsW', 'value'].iloc[0])
    dCsW = np.arctanh(summary_metrics_df.loc[summary_metrics_df['metric']=='dCsW', 'value'].iloc[0])
    sCdW = np.arctanh(summary_metrics_df.loc[summary_metrics_df['metric']=='sCdW', 'value'].iloc[0])
    dCdW = np.arctanh(summary_metrics_df.loc[summary_metrics_df['metric']=='dCdW', 'value'].iloc[0])
    
    # Calculate contrasts and apply inverse Fisher transform
    contrast_metrics_dict['sCsW-dCdW'] = np.tanh(sCsW - dCdW)
    contrast_metrics_dict['dCsW-dCdW'] = np.tanh(dCsW - dCdW)
    contrast_metrics_dict['sCdW-dCdW'] = np.tanh(sCdW - dCdW)
    contrast_metrics_dict['dCsW-sCdW'] = np.tanh(dCsW - sCdW)
    contrast_metrics_dict['sCsW-dCsW'] = np.tanh(sCsW - dCsW)
    contrast_metrics_dict['sCsW-sCdW'] = np.tanh(sCsW - sCdW)
    
    return contrast_metrics_dict

def rename_columns_and_index(df, dict_c1, dict_c2):
    # Create a function to process each column/index name
    def process_name(name):
        # Split the name into year and remainder
        year, rest = name.split('_', 1)
        
        # Choose appropriate dictionary based on whether it's C1 or C2
        if 'C1' in rest:
            new_year = dict_c1[year]
        else:  # C2
            new_year = dict_c2[year]
            
        # Return the new name with the year replaced
        return f"{new_year}_{rest}"
    
    # Create new column names
    new_columns = [process_name(col) for col in df.columns]
    
    # Create new index names
    new_index = [process_name(idx) for idx in df.index]
    
    # Rename both columns and index
    df = df.rename(columns=dict(zip(df.columns, new_columns)))
    df = df.rename(index=dict(zip(df.index, new_index)))
    
    return df

def plot_correlation_matrix(reduced_results, area_name, resultsLoc):
    """Plot correlation matrix for reduced results (3x6 matrix)"""
    # Use the transposed matrix directly
    corr_matrix = reduced_results.T.astype(float).values
    
    # Get rounded ceiling for cmap
    matrix_ceil = np.ceil(np.abs(reduced_result.values).max() / 0.1) * 0.1

    # Set up figure and font sizes
    plt.figure(figsize=(2.4, 1.2))
    SMALL_SIZE = 6
    MEDIUM_SIZE = 6
    BIGGER_SIZE = 6
    
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    # Plot correlation matrix
    im = plt.imshow(corr_matrix, aspect='auto', cmap='PRGn')
    cbar = plt.colorbar()
    #plt.clim(matrix_ceil*-1, matrix_ceil)
    plt.clim(-0.5, 0.5)
    cbar.ax.tick_params(labelsize=SMALL_SIZE)
    
    # Add vertical line to separate C1 and C2 windows
    plt.axvline(x=2.5, color='black', linewidth=1.0)
    
    # Set ticks and labels
    x_labels = ['Early\nC1', 'Late\nC1', 'Go\nC1', 
                'Early\nC2', 'Late\nC2', 'Go\nC2']
    y_labels = ['Early C2', 'Late C2', 'Go C2']
    
    plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(y_labels)), y_labels)

    # Adjust layout
    plt.tight_layout()

    # Font settings for SVG
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'arial'

    # Save figures
    plt.savefig(resultsLoc + 'print/matrix_{0}.svg'.format(area_name), 
                format='svg', dpi=300, bbox_inches="tight")
    plt.savefig(resultsLoc + 'matrix_{0}.png'.format(area_name), 
                dpi=300, bbox_inches="tight")
    plt.close()

#### Execution
print('fsSpikeTaskphaseDecodeV.py')
area_num = parse_args()
area_info = areas_sweep[area_num]
#area_info = areas_sweep[2]
area_list = area_info[0]
analysisdf = analysisdf_import.loc[analysisdf_import['area'].isin(area_list)].reset_index(drop=True)
cv_results_frame = create_cv_results_frame(analysisdf, resultsLoc, path_group_combinations, area_info, analysisdf['Unit'].unique(), grouped_projection_windows)

### Tools and functions needed for processing

# Create dictionaries to rename columns / indices
C1_rename = {}
for i_path in pathway_numbers:
    curr_mazes_row = mazes.loc[(mazes['Npath']==i_path)&(mazes['ChoiceNo']=='ChoiceI')]
    C1_rename[str(i_path)] = '{}_{}_{}'.format(
        str(curr_mazes_row['FP'].iloc[0]),
        str(curr_mazes_row['NextFPmap'].iloc[0]),
        str(curr_mazes_row['goal'].iloc[0]))
C2_rename = {}
for i_path in pathway_numbers:
    curr_mazes_row = mazes.loc[(mazes['Npath']==i_path)&(mazes['ChoiceNo']=='ChoiceII')]
    C2_rename[str(i_path)] = '{}_{}_{}'.format(
        str(curr_mazes_row['FP'].iloc[0]),
        str(curr_mazes_row['NextFPmap'].iloc[0]),
        str(curr_mazes_row['goal'].iloc[0]))

## Function to extract the three leading indicators
def get_indicators(name):
    # Split by underscore and take first three elements
    return name.split('_')[:3]  # Returns a list of the three indicators

# Function to get window name (everything after the last underscore)
def get_window_name(name):
    return name.split('_')[-1]

def reduce_cv_results(cv_results_frame, grouped_projection_windows):
    # Remove field with overlapping indicators
    # Loop through both dimensions of the dataframe
    for col in cv_results_frame.columns:
        col_ind1, col_ind2, col_ind3 = get_indicators(col)
        
        for idx in cv_results_frame.index:
            idx_ind1, idx_ind2, idx_ind3 = get_indicators(idx)
            
            # If any of the indicators match, set value to np.nan
            if (col_ind1 == idx_ind1 or 
                col_ind2 == idx_ind2 or 
                col_ind3 == idx_ind3):
                cv_results_frame.loc[idx, col] = np.nan

    # Initialize the result DataFrame with the known window names
    reduced_result = pd.DataFrame(index=grouped_projection_windows.keys(), 
                        columns=grouped_projection_windows.keys())

    # For each combination of window names
    for row_window in grouped_projection_windows.keys():
        for col_window in grouped_projection_windows.keys():
            # Get all columns and rows that end with these window names
            relevant_cols = [col for col in cv_results_frame.columns if get_window_name(col) == col_window]
            relevant_rows = [idx for idx in cv_results_frame.index if get_window_name(idx) == row_window]
            
            # Get all values for this window combination
            values = cv_results_frame.loc[relevant_rows, relevant_cols].values.flatten()
            
            # Remove NaN values
            values = values[~np.isnan(values)]
            
            if len(values) > 0:
                # Apply Fisher transformation for correlation averaging
                transformed_values = np.arctanh(values)
                mean_transformed = np.mean(transformed_values)
                mean_correlation = np.tanh(mean_transformed)
                reduced_result.loc[row_window, col_window] = mean_correlation
            else:
                reduced_result.loc[row_window, col_window] = np.nan
    reduced_result = reduced_result.dropna(axis=1)
    return reduced_result

### Continue pipeline

# Rename columns and index
cv_results_frame = rename_columns_and_index(cv_results_frame, C1_rename, C2_rename)
cv_results_frame_backup = cv_results_frame.copy()

# Reduce the cv_results_frame
reduced_result = reduce_cv_results(cv_results_frame, grouped_projection_windows)

# Save the reduced result to disk
reduced_result.to_csv(resultsLoc+area_info[1]+'_reduced_result.csv')
plot_correlation_matrix(reduced_result, area_info[1], resultsLoc)

# Create summary metrics
summary_metrics_df = create_summary_metrics(reduced_result, area_info[1], create_figures = True)
contrast_metrics_dict = calculate_contrast_metrics(summary_metrics_df)

# Add additional contrast for baseline comparison
## early choice C2 with go C1 vs early choice C2 with go C2
baseline_valI = reduced_result.loc['GoC1', 'EarlyC2']
baseline_valII = reduced_result.loc['GoC2', 'EarlyC2']
baseline_contrast = np.tanh(np.arctanh(baseline_valI) - np.arctanh(baseline_valII))
contrast_metrics_dict['GoC1xEarlyC2-GoC2xEarlyC2'] = baseline_contrast

# Create df based on contrast_metrics_dict, with additional column for p-values
contrast_metrics_df = pd.DataFrame.from_dict(contrast_metrics_dict, orient='index', columns=['value'])
contrast_metrics_df['p_value'] = np.nan

# Create dict with same keys but empty lists as values
contrast_dict_bootstrap = {}
for i_key in contrast_metrics_dict.keys():
    contrast_dict_bootstrap[i_key] = []

# Created bootstrapped projections
analysisdf.reset_index(inplace=True)

for i_split in range(1000):
    # Resample data
    print(f"Fold {i_split}:")
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
    cv_results_frame_boot = rename_columns_and_index(cv_results_frame_boot, C1_rename, C2_rename)
    cv_results_frame_boot_backup = cv_results_frame_boot.copy()

    # Reduce the cv_results_frame
    cv_results_frame_boot = reduce_cv_results(cv_results_frame_boot, grouped_projection_windows)

    # Create summary metrics
    summary_metrics_df_boot = create_summary_metrics(cv_results_frame_boot, area_info[1], create_figures = False)
    contrast_metrics_dict_boot = calculate_contrast_metrics(summary_metrics_df_boot)

    # Add baseline contrast
    baseline_valI = cv_results_frame_boot.loc['GoC1', 'EarlyC2']
    baseline_valII = cv_results_frame_boot.loc['GoC2', 'EarlyC2']
    baseline_contrast = np.tanh(np.arctanh(baseline_valI) - np.arctanh(baseline_valII))
    contrast_metrics_dict_boot['GoC1xEarlyC2-GoC2xEarlyC2'] = baseline_contrast

    ## Add each list from results_dict to boot_results_dict
    for key, value in contrast_metrics_dict_boot.items():
        contrast_dict_bootstrap[key].append(value)

# Extract p_values
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