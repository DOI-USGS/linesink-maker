import sys
sys.path.insert(0, '../linesinkmaker')
import os
import numpy as np
from shapely.geometry import LineString
import lsmaker

def run_test():
    if not os.path.isdir('output'):
        os.makedirs('output')
    input_file = 'test_input.xml'
    ls = lsmaker.linesinks(input_file)

    #verify that input is being read correctly
    nearfield_tol = 50
    routed_area_tol = 100
    farfield_tol = 200
    assert ls.nearfield == 'input/testnearfield.shp'
    assert ls.farfield_buffer == 1000
    assert ls.nearfield_tolerance == nearfield_tol
    assert ls.routed_area_tolerance == routed_area_tol
    assert ls.farfield_tolerance == farfield_tol
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
    i = ls.df.GNIS_NAME == 'Chequamegon Waters 125 Reservoir'
    assert ls.df.ix[i, 'dncomid'].values[0][0] == 13103287
    assert np.abs(ls.df.ix[i, 'maxElev'].values[0] - 1256.85) < 0.01
    assert ls.df.ix[i, 'routing'].values[0] == 1

    assert 'Lost Lake' in ls.df.GNIS_NAME.values
    i = ls.df.GNIS_NAME == 'Lost Lake'
    assert ls.df.ix[i, 'farfield'].values[0] # this one should be in the farfield
    assert np.abs(ls.df.ix[i, 'maxElev'].values[0] - 1330.05) < 0.01
    assert ls.df.ix[i, 'routing'].values[0] == 0

    # 585 lines should be written
    # sum with nf at 50, routed area at 100, ff at 200, dropping ephemeral streams
    #assert sum([len(p) - 1 for p in ls.df.ls_coords]) == 684
    # sum with whole routed area at 100, ff at 200:
    #assert sum([len(p) - 1 for p in ls.df.ls_coords]) == 578
    #assert sum([len(p) - 1 for p in ls.df.ls_coords]) == 585 # number of lines with single nearfield

    # check for correct level of simplification
    def check_simplification(df, i, tol):
        orig = np.array(df.geometry[i].values[0].coords)
        simp = np.array(df[i].ls_coords.values[0])
        simp2 = np.array(LineString(orig).simplify(tol).coords)
        assert np.array_equal(simp, simp2)

    check_simplification(ls.df,
                         ls.df.COMID == 13102931, # Johns Creek
                         nearfield_tol)
    check_simplification(ls.df,
                         ls.df.COMID == 13103273, #South Fork Yellow River
                         routed_area_tol)
    check_simplification(ls.df,
                         ls.df.COMID == 13132153, #Black River
                         farfield_tol)



def run_test2():
    if not os.path.isdir('output'):
        os.makedirs('output')
    input_file = 'test_input.xml'
    ls = lsmaker.linesinks(input_file)
    ls.nearfield = 'input/testarea.shp'
    ls.routed_area = None
    ls.routed_area_tolerance = None
    ls.drop_intermittent = False
    ls.nearfield_tolerance = 100
    ls.farfield_tolerance = 200
    ls.outfile_basename = 'output/test2'

    ls.preprocess(save=True)

    ls.makeLineSinks()

    assert sum([len(p) - 1 for p in ls.df.ls_coords]) == 586 # not sure why this is 586 vs. 585 above
    # likewise number above changed from 683 to 684 after debugging

if __name__ == '__main__':
    run_test()
    run_test2()
