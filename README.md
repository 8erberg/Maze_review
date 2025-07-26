# Code relating to 'Neural dynamics of an extended frontal lobe network in goal-subgoal problem solving' 

*Valentina Mione, Jascha Achterberg, Makoto Kusunoki, Mark J. Buckley, and John Duncan*

The manuscript can be found on: https://www.biorxiv.org/content/10.1101/2025.05.06.652442v1.abstract 

## Overview of analyses scripts

The following table lists which analysis and preprocessing scripts are used to generate a given figure. Note that PreAnalysis scripts are run once in isolation to preprocess the data and are later used for importing the preprocessed data. Below we give detailed instructions on how to execute a given script.  

| Figure / analyses       | Preprocessing scripts ('PreAnalysis')                                     | Analysis scripts                 |
|-------------------------|---------------------------------------------------------------------------|----------------------------------|
| Figure 2                | -                                                                         | Figure2_PEVstats_fdr.m           |
| Figure 3                | fsSpike_GallC_100ms_StimOn_Go_II.py                                       | dimsSpikeGoalI_2ndPopulation.py  |
| Figure 4                | fsSpike_GallC_100ms_StimOn_Go_II.py                                       | dimsSpikeGoalPositionMoveI.py    |
| Figure 5                | fsSpike_GallC_100ms_StimOn_Go_II.py                                       | dimsSpikeMoveI_2ndPopulation.py  |
| Figure 6                | fsSpike_GC1C2_50ms_StimOn_Go_I.py                                         | dimsSpikeC1II.py                 |
| Figure 7                | fsSpike_GallC_100ms_StimOn_Go_II.py                                       | fsSpikeTaskphaseDecodeIV_boot.py |
| Figure S1               | fsSpike_GallC_100ms_StimOn_Go_II.py                                       | dimsSpikeGoalI_2ndPopulation.py  |
| Figure S2               | fsSpike_GC1C2_50ms_StimOn_Go_I.py                                         | dimsSpikeC1II_4step.py           |
| Figure S3               | fsSpike_GallC_100ms_StimOn_Go_II.py                                       | fsSpikeTaskphaseDecodeV.py       |
| Orthogonality of spaces | fsSpike_GallC_100ms_StimOn_Go_II.py                                       | dimsSpikeCrossDimsProjI.py       |


## System requirements and runtime

All Python-based analyses are based on the conda environment specified in [conda_env.txt](https://github.com/8erberg/Maze_review/blob/main/conda_env.txt). We execute these on a HPC system running Ubuntu 24.04.1 LTS, Slurm 23.11.4 and conda 23.11.0, equipped with AMD EPYC 7302 16-Core Processor with 20GB of RAM allocated for job execution. All scripts are written so that they can make use of parallel execution across multiple jobs allocated to separate cores / nodes if a run across the entire data would otherwise require >6 hours of runtime. When run in parallel, any single job should hence complete in <6 hours on a comparable HPC system. Preprocessing scripts are executed 'per recording session' during preprocessing, and all analysis scripts are executed 'per analysed brain region', with the exception of [dimsSpikeCrossDimsProjI.py](https://github.com/8erberg/Maze_review/blob/main/Scripts/Dimensions/dimsSpikeCrossDimsProjI.py) which is executed across all regions simultaneously as it is lightweight with a short runtime. Execution time may vary from ours, especially on non x86 hardware outside HPC environments. 

## How to run analyses

In the following we specify how to run our analyses. Each analysis script directly generates the figures which are then used in the manuscript. Note that our execution calls assume a Slurm-based HPC system.

### Installation

Clone this repository, then from within the repository run:
```
conda create --name your_environment_name conda_env.txt
```

Note that most preprocessing and analysis scripts (like https://github.com/8erberg/Maze_review/blob/main/Scripts/PreAnalysis/fsSpike_GC1C2_50ms_StimOn_Go_I.py#L16) need you to manually add your system path of the 'Scripts' folder contained in this repository in the following line of each script:
```
sys.path.append('...') # Should be the path of .../Maze_review/Scripts/
```

### Download and preprocess data

We provide the entire spike sorted dataset for review so that analysis scripts can be executed truthfully. To download the dataset, follow the instructions given here: https://github.com/8erberg/Maze_review/blob/main/Files/Data/Spikes/Download_instructions.md

Once all data has been downloaded into the respective folder https://github.com/8erberg/Maze_review/tree/main/Files/Data/Spikes, the data needs to be preprocessed. Preprocessing scripts are contained in the folder https://github.com/8erberg/Maze_review/tree/main/Scripts/PreAnalysis and can be executed with the provided execution script https://github.com/8erberg/Maze_review/blob/main/Scripts/PreAnalysis/run.py (note that you can comment / uncomment the respective script name in https://github.com/8erberg/Maze_review/blob/main/Scripts/PreAnalysis/run.py#L5 to select which preprocessing routine to run). To execute the preprocessing of the data, simply navigate to the PreAnalysis folder https://github.com/8erberg/Maze_review/tree/main/Scripts/PreAnalysis and run:
```
conda activate your_environment_name
python run.py
```
This will preprocess the data by scheduling individual jobs via Slurm, one per recording session.

### Executing analysis script

Before executing an analysis script, create the respective results folder like outlined in https://github.com/8erberg/Maze_review/blob/main/Files/Results/Note_on_results_folder.md . Otherwise executing analysis scripts follows the same logic as for preprocessing scripts. Navigate to the folder of the analysis script you want to run, uncomment the relevant script ('executed_script_name') in https://github.com/8erberg/Maze_review/blob/main/Scripts/Dimensions/run.py or https://github.com/8erberg/Maze_review/blob/main/Scripts/FullSetExplore/run.py and then run:
```
conda activate your_environment_name
python run.py
```

This will run analyses across multiple slurm jobs. Note that dimsSpikeCrossDimsProjI.py is an exception and is executed as detailed in https://github.com/8erberg/Maze_review/blob/main/Scripts/Dimensions/run.py#L10 .

## License
MIT License - Copyright (c) 2025 MRC Cognition and Brain Sciences Unit
