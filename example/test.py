__author__ = 'aleaf'

import sys
#sys.path.append('D:/ATLData/Documents/GitHub/GFLOW_utils')
sys.path.append('..')
import lsmaker as lsm
import GISio
from diagnostics import *
import os


ls = lsm.linesinks('Nicolet_lines.xml')

ls.preprocess_arcpy()

ls.preprocess()

ls.makeLineSinks(shp='preprocessed/lines.shp')

ls.run_diagnostics()

