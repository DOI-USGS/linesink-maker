import sys
sys.path.insert(0, '..')
import warnings
import os
import numpy as np
from shapely.geometry import LineString, Point
import pytest
from lsmaker.lsmaker import LinesinkData
from lsmaker import get_elevation_from_epqs, get_elevations_from_epqs


@pytest.fixture()
def preprocessed_folder():
    # pathing to preprocessed files not totally flexible yet
    # need to write to /preprocessed relative to config file
    preprocessed_folder = 'lsmaker/tests/data/preprocessed'
    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder)
    return preprocessed_folder


def test1(test_data_path, test_output_path, preprocessed_folder):

    input_file = os.path.join(test_data_path, 'test_input.xml')
    ls = LinesinkData(input_file)

    # verify that input is being read correctly
    nearfield_tol = 50
    routed_area_tol = 100
    farfield_tol = 200
    expected_nearfield_path = os.path.abspath(os.path.join(ls._lsmaker_config_file_path,
                                                           'testnearfield.shp'))
    assert ls.nearfield == expected_nearfield_path
    assert ls.farfield_buffer == 1000
    assert ls.nearfield_tolerance == nearfield_tol
    assert ls.routed_area_tolerance == routed_area_tol
    assert ls.farfield_tolerance == farfield_tol
    assert ls.min_farfield_order == 2
    assert ls.min_waterbody_size == 0.001
    assert ls.min_farfield_wb_size == 0.001

    ls.preprocess(save=True)

    ls.make_linesinks(shp=os.path.join(preprocessed_folder, 'lines.shp'))

    # verify that drainage lakes are included properly
    assert 'Chequamegon Waters 125 Reservoir' in ls.df.GNIS_NAME.values

    # test that lake has correct COMID and all upstream tribs are accounted for
    assert ls.df.loc[ls.df.GNIS_NAME == 'Chequamegon Waters 125 Reservoir', 'COMID'].values[0] == '13102555'
    assert np.array_equal(
            np.array(sorted(ls.df.loc[ls.df.GNIS_NAME == 'Chequamegon Waters 125 Reservoir', 'upcomids'].values[0])),
            np.array(sorted(['13102899', '13102915', '13103263', '13102907', '13102927'])))

    # test for correct routing; also tests that dncomids are lists (instead of integers)
    i = ls.df.GNIS_NAME == 'Chequamegon Waters 125 Reservoir'
    assert ls.df.loc[i, 'dncomid'].values[0][0] == '13103287'
    assert np.abs(ls.df.loc[i, 'maxElev'].values[0] - 1256.85) < 0.01
    assert ls.df.loc[i, 'routing'].values[0] == 1

    assert 'Lost Lake' in ls.df.GNIS_NAME.values
    i = ls.df.GNIS_NAME == 'Lost Lake'
    assert ls.df.loc[i, 'farfield'].values[0] # this one should be in the farfield
    assert np.abs(ls.df.loc[i, 'maxElev'].values[0] - 1330.05) < 0.01
    assert ls.df.loc[i, 'routing'].values[0] == 0

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
                         ls.df.COMID == '13102931', # Johns Creek
                         nearfield_tol)
    check_simplification(ls.df,
                         ls.df.COMID == '13103273', #South Fork Yellow River
                         routed_area_tol)
    check_simplification(ls.df,
                         ls.df.COMID == '13132153', #Black River
                         farfield_tol)


@pytest.mark.parametrize(('case', 'min_waterbody_size', 'expected_nlines'), 
                         ((1, 1e10, 412),
                          (2, 0.001, 586)))
def test2(test_data_path, test_output_path, case, min_waterbody_size, expected_nlines):
    input_file = os.path.join(test_data_path, 'test_input.xml')
    ls = LinesinkData(input_file)
    ls.nearfield = os.path.join(test_data_path, 'testarea.shp')
    ls.routed_area = None
    ls.routed_area_tolerance = None
    ls.drop_intermittent = False
    ls.nearfield_tolerance = 100
    ls.farfield_tolerance = 200
    ls.outfile_basename = os.path.join(test_output_path, 'test2')
    ls.min_waterbody_size = min_waterbody_size
    ls.min_nearfield_wb_size = min_waterbody_size
    ls.min_farfield_wb_size = min_waterbody_size

    ls.preprocess(save=True)

    ls.make_linesinks()

    assert sum([len(p) - 1 for p in ls.df.ls_coords]) == expected_nlines # not sure why this is 586 vs. 585 above
    # likewise number above changed from 683 to 684 after debugging


def test_epqs():
    elev = get_elevation_from_epqs(-91.5, 46.8, units='Feet')
    elev = float(elev)
    if np.abs(602-elev) > 1:
        warnings.warn('Bad elevation value of {}'.format(elev))
    elevs = get_elevations_from_epqs([Point(-91.5, 46.8)] * 2, units='Feet')
