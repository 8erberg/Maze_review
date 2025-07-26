#!/usr/bin/env python
# -*- coding: utf-8 -*-

executed_script_name = 'fsSpikeTaskphaseDecodeIV_boot.py'
#executed_script_name = 'fsSpikeTaskphaseDecodeV.py'

import sys
from subprocess import call
sys.path.append('...')

areas = range(6)
for i_area in areas:
    cmd = "#!/bin/bash \npython {0} -area_num {1}".format(executed_script_name, str(i_area)) # form the command
    file = open(str(i_area) + '_script.sh', 'w') # place the command within a .sh file, and this defines the name of this file which will run on the cluster
    file.write(cmd)
    file.close()
    cmd = "sbatch --time=3-00:00:00 --mem-per-cpu=20G --mincpus=1 " + str(i_area) + "_script.sh" # run command on the cluster, with the sbatch settings
    call(cmd, shell=True)