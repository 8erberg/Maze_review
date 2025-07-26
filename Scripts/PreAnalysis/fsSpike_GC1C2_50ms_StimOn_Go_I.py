#!/usr/bin/env python
# -*- coding: utf-8 -*-

# File-Name:    fsSpike_GC1C2_50ms_StimOn_Go_I.py
# Version:      0.0.3

# Import packages
import sys
import pandas as pd
import numpy as np
import argparse
from glob import glob
import os

# Import own packages
sys.path.append('...')
from Functions.Import.LocationsI import spikesDataII, resultsFolderI
from Functions.Preprocessing.dictionariesI import monkeyNameDicI, arealDicII

# Locations
resultsLoc = resultsFolderI+"PreAnalysis/fsSpike_GC1C2_50ms_StimOn_Go_I/"
import_loc = spikesDataII+'230413/'

# Define window characteristics
anchortimes = {'UNFIXTG11_ON':[-0.3,0.35,14], 
               'FIXTG0_OFF':[-0.2,0.05,6],
               'UNFIXTG21_ON':[-0.4,0.35,16],
               'FIXTG11_OFF':[-0.2,0.05,6],
               'UNFIXTG41_ON':[0.2,0.35,4],
               'FIXTG31_OFF':[0.0,0.05,2]}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_name", type=str, 
                        help="Session name for import / preprocessing")
    args = parser.parse_args()
    return args

def resolve_anchortimes(anchortimes_dict, spikerow):
    timewindows = []
    timewindows_names = []
    for i_anchortime, i_anchorvars in anchortimes_dict.items():
        vecstarts = np.linspace(start=i_anchorvars[0], stop=i_anchorvars[1], num=i_anchorvars[2])
        vecends = np.linspace(start=i_anchorvars[0]+0.05, stop=i_anchorvars[1]+0.05, num=i_anchorvars[2])
        vecstarts = np.round(vecstarts, 2)
        vecends = np.round(vecends, 2)
        for i_window in range(len(vecstarts)):
            timewindows.append((spikerow[i_anchortime]+vecstarts[i_window], spikerow[i_anchortime]+vecends[i_window]))
            timewindows_names.append(i_anchortime+'_'+str(vecstarts[i_window]))
    
    return timewindows, timewindows_names

def create_time_window_names(anchortimes_dict):
    timewindows_names = []
    for i_anchortime, i_anchorvars in anchortimes_dict.items():
        vecstarts = np.linspace(start=i_anchorvars[0], stop=i_anchorvars[1], num=i_anchorvars[2])
        vecstarts = np.round(vecstarts, 2)
        for i_window in range(len(vecstarts)):
            timewindows_names.append(i_anchortime+'_'+str(vecstarts[i_window]))
    
    return timewindows_names

def spike_process_session(session_name):
    #curr_session = 'v081119_s'
    #curr_session = 'u101221t_s'
    curr_session = session_name
    
    # Extract monkey name from session name
    monkey_name = monkeyNameDicI[curr_session[0]]
    
    # Import metadata
    metadata = pd.read_csv(import_loc+'MetaData_{0}.csv'.format(monkey_name))

    # Select correct 1st choice trials from example sessions
    metadata = metadata.loc[
        (metadata['session']==curr_session)&
        (metadata['Choice1step']==1)
    ]

    # Drop trials where the choice wasnt correctly completed
    metadata = metadata.dropna(subset=['START_FIX1'])

    # Read spikeframe
    spikeframe = pd.read_csv(import_loc+'Spikes_{0}.csv'.format(monkey_name))
    spikeframe = spikeframe.loc[spikeframe['session']==curr_session]

    # Create list of spike counts for each NeuronXTrial
    countlists = []
    for i in range(len(metadata)):
        ## Extract spike train
        spikerow = metadata.iloc[i]
        spikearray = spikeframe.loc[
            (spikeframe['NeuronNum']==spikerow.NeuronNum)&
            (spikeframe['TrialNum']==spikerow.TrialNum)] 
        # Create array with only the spike times
        spikearray = spikearray.drop(columns=['session', 'NeuronNum', 'NeuronLabel', 'unit_id', 'rec_area','sorting_score', 'TrialNum']).iloc[0].dropna().to_numpy()

        ## Resolve multiple anchortimes and timewindows to spike times
        timewindows, timewindows_names = resolve_anchortimes(anchortimes, spikerow)

        ## Count spikes in each window
        windowlist = []
        for i_window in np.arange(len(timewindows_names)):
            curr_start = timewindows[i_window][0]
            curr_end = timewindows[i_window][1]
            if np.isnan(curr_start):
                temp = np.nan
            else:
                temp = len(np.where((spikearray>curr_start)&(spikearray<curr_end))[0])
            windowlist.append(temp)
        countlists.append(windowlist)

    # Add countlists as columns to dataframe
    count_df = pd.DataFrame(countlists, columns = timewindows_names)
    analysisdf = pd.concat([metadata.reset_index(), count_df], axis=1)

    # Save analysisdf
    save_analysisdf = analysisdf[['session','NeuronNum','TrialNum','FinalPos','rec_area','PathNum','Choice1step','Choice2step','Choice3step','Choice4step']+timewindows_names]
    save_analysisdf.to_csv(resultsLoc+'{0}_spikecounts.csv'.format(curr_session), index=False)

def import_processed_spike_data():
    # Import all .csv files in Precompute folder
    all_files = glob(os.path.join(resultsLoc, "*spikecounts.csv"))
    def open_file(f):
        temp = pd.read_csv(f)
        return temp
    df_from_each_file = (open_file(f) for f in all_files)
    analysisdf = pd.concat(df_from_each_file, ignore_index=True)
    analysisdf['area'] = analysisdf['rec_area'].map(arealDicII)
    analysisdf['PathNum'] = analysisdf['PathNum']-9100
    return analysisdf

if __name__ == "__main__":
    args = parse_args()
    spike_process_session(args.session_name)
    print('Finished: {0}'.format(args.session_name))
