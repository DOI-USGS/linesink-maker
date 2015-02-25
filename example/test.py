__author__ = 'aleaf'

import sys
#sys.path.append('D:/ATLData/Documents/GitHub/GFLOW_utils')
sys.path.append('..')
import lsmaker as lsm
import GISio
from diagnostics import *

import os
print os.getcwd()

ls = lsm.linesinks('Nicolet_lines.xml')

#ls.preprocess()

ls.makeLineSinks(shp='preprocessed/lines.shp')
'''
dg = Diagnostics(lsm_object=ls)
dg.check_vertices()
dg.check4crossing_lines()
dg.check4zero_gradient()
'''
