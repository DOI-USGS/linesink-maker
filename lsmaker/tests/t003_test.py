"""
Run Medford test example
"""
import sys
sys.path.insert(0, '..')
import lsmaker
import os


def test_medford():

    path = 'examples/example_Medford/Medford_lines.xml'
    dir, input_file = os.path.split(path)

    os.chdir(dir)
    ls = lsmaker.linesinks(input_file)

    ls.preprocess(save=True)

    ls.makeLineSinks(shp='preprocessed/lines.shp')