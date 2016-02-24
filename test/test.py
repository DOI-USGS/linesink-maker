import sys
sys.path.insert(0, '../linesinkmaker')
import lsmaker

input_file = 'test_input.xml'
ls = lsmaker.linesinks(input_file)

ls.preprocess(save=True)

ls.makeLineSinks(shp='preprocessed/lines.shp')
