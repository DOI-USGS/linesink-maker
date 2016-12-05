import sys
sys.path.insert(0, '..')
import warnings
import os
import numpy as np
from shapely.geometry import LineString, Point
import lsmaker
from lsmaker import get_elevation_from_epqs, get_elevations_from_epqs

def test_imports():
    from lsmaker import GISio
    from lsmaker.utils.GFLOWresults import write_streamflow_shapefile

def test_deps():
    """test the dependencies"""
    try:
        # For Python 3.0 and later
        from urllib.request import urlopen
    except ImportError:
        # Fall back to Python 2's urllib2
        from urllib2 import urlopen

    import requests
    import json
    from functools import partial
    import fiona
    from fiona.crs import to_string
    from shapely.geometry import Polygon, LineString, Point, shape, mapping
    from shapely.ops import unary_union, transform
    import pyproj
    import math

    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    output = fiona.open('tmp/test.shp', 'w',
                        crs={'init': 'epsg:26715'},
                        schema={'geometry': 'Point',
                                'properties': {'Id': 'int:6'}},
                        driver='ESRI Shapefile')
    output.write({'properties': {'Id': 0},
                  'geometry': mapping(Point(0, 0))})
    output.close()

    # verify that projection file was written
    assert os.stat('tmp/test.prj').st_size > 0

    shp_obj = fiona.open('tmp/test.shp')
    shp_obj.schema

def test1():
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



def test2():
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

def test_epqs():
    elev = get_elevation_from_epqs(-91.5, 46.8, units='Feet')
    elev = float(elev)
    if np.abs(602-elev) > 1:
        warnings.warn('Bad elevation value of {}'.format(elev))
    elevs = get_elevations_from_epqs([Point(-91.5, 46.8)] * 100, units='Feet')

if __name__ == '__main__':
    test_imports()
    test_deps()
    test_epqs()
    test1()
    test2()

