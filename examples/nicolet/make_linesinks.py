import sys
import lsmaker

try:
    input_file = sys.argv[1]
except IndexError:
    print("\nusage is: python make_linesinks.py <configuration file>\n")
    quit()

ls = lsmaker.LinesinkData(input_file)

ls.preprocess(save=True)

ls.make_linesinks(shp='preprocessed/lines.shp')
