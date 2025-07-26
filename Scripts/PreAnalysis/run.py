#!/usr/bin/env python
# -*- coding: utf-8 -*-

executed_script_name = 'fsSpike_GC1C2_50ms_StimOn_Go_I.py'
#executed_script_name = 'fsSpike_GallC_100ms_StimOn_Go_II.py'

import sys
from subprocess import call
sys.path.append('...')
from Functions.Import.LocationsI import spikesDataII

# Import available spiking sessions in current set of data
import_loc = spikesDataII+'230413/'
session_import = []
with open(import_loc+'sessionlist_viking.txt', 'r') as f:
    sessionlist = f.read().split()
session_import = session_import + sessionlist
with open(import_loc+'sessionlist_uriel.txt', 'r') as f:
    sessionlist = f.read().split()
session_import = session_import + sessionlist

# Execute scripts over sessions
for session in session_import:
    cmd = "#!/bin/bash \npython {0} {1}".format(executed_script_name, session) # form the command
    file = open(session + '_script.sh', 'w') # place the command within a .sh file, and this defines the name of this file which will run on the cluster
    file.write(cmd)
    file.close()
    cmd = "sbatch --time=3-00:00:00 --mem-per-cpu=20G --mincpus=1 " + session + "_script.sh" # run command on the cluster, with the sbatch settings
    call(cmd, shell=True)
