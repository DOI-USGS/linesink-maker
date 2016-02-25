import sys
sys.path.insert(0, '../linesinkmaker')
import os
import numpy as np
import lsmaker

def run_test():
    if not os.path.isdir('output'):
        os.makedirs('output')
    input_file = 'test_input.xml'
    ls = lsmaker.linesinks(input_file)

    #verify that input is being read correctly
    assert ls.nearfield == 'input/testnearfield.shp'
    assert ls.farfield_buffer == 1000
    assert ls.nearfield_tolerance == 50
    assert ls.farfield_tolerance == 200
    assert ls.min_farfield_order == 2
    assert ls.min_waterbody_size == 0.001
    assert ls.min_farfield_wb_size == 0.001

    ls.preprocess(save=True)

    ls.makeLineSinks(shp='preprocessed/lines.shp')

    # verify that drainage lakes are included properly
    assert 'Chequamegon Waters 125 Reservoir' in ls.df.GNIS_NAME.values

    # test that lake has correct COMID and all upstream tribs are accounted for
    assert ls.df.ix[ls.df.GNIS_NAME == 'Chequamegon Waters 125 Reservoir', 'COMID'].values[0] == 13102555
    assert np.array_equal(
            np.array(sorted(ls.df.ix[ls.df.GNIS_NAME == 'Chequamegon Waters 125 Reservoir', 'upcomids'].values[0])),
            np.array(sorted([13102899, 13102915, 13103263, 13102907, 13102927])))
    # test for correct routing; also tests that dncomids are lists (instead of integers)
    assert ls.df.ix[ls.df.GNIS_NAME == 'Chequamegon Waters 125 Reservoir', 'dncomid'].values[0][0] == 13103287
    assert np.abs(ls.df.ix[ls.df.GNIS_NAME == 'Chequamegon Waters 125 Reservoir', 'maxElev'].values[0] - 1256.85) < 0.01
    assert 'Lost Lake' in ls.df.GNIS_NAME.values
    assert ls.df.ix[ls.df.GNIS_NAME == 'Lost Lake', 'farfield'].values[0] # this one should be in the farfield
    assert np.abs(ls.df.ix[ls.df.GNIS_NAME == 'Lost Lake', 'maxElev'].values[0] - 1330.05) < 0.01

    # 585 lines should be written
    # sum with nf at 50, routed area at 100, ff at 200, dropping ephemeral streams
    assert sum([len(p) - 1 for p in ls.df.ls_coords]) == 683
    # sum with whole routed area at 100, ff at 200:
    #assert sum([len(p) - 1 for p in ls.df.ls_coords]) == 578
    #assert sum([len(p) - 1 for p in ls.df.ls_coords]) == 585 # number of lines with single nearfield


if __name__ == '__main__':
    run_test()
