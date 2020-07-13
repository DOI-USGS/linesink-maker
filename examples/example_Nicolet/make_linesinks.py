
# coding: utf-8

# In[1]:

import sys
sys.path.insert(0, '..')
import lsmaker

try:
    input_file = sys.argv[1]
except IndexError:
    print("\nusage is: python make_linesinks.py <input_xml_file>\n")
    quit()

ls = lsmaker.linesinks(input_file)

ls.preprocess(save=True)

ls.makeLineSinks(shp='preprocessed/lines.shp')
