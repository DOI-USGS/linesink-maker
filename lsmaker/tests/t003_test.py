"""Tests based on the Medford example."""
import lsmaker
import os
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import LineString
import pytest
import subprocess


@pytest.fixture(scope='module')
def lsmaker_instance_from_xml():
    config_file = 'examples/medford/Medford_lines.xml'
    ls = lsmaker.LinesinkData(config_file)
    return ls


@pytest.fixture(scope='module')
def lsmaker_instance_with_linesinks(lsmaker_instance_from_xml):
    ls = lsmaker_instance_from_xml
    ls.make_linesinks()
    return ls


def test_medford(lsmaker_instance_with_linesinks):
    ls = lsmaker_instance_with_linesinks
    assert isinstance(ls, lsmaker.LinesinkData)
    
    # check tolerances that were applied to each linesink
    assert all(ls.df.loc[~ls.df['routed'] & ls.df['nearfield'], 'tol'] == ls.nearfield_tolerance)
    assert all(ls.df.loc[ls.df['routed'] & ~ls.df['nearfield'], 'tol'] == ls.routed_area_tolerance)
    assert all(ls.df.loc[ls.df['farfield'], 'tol'] == ls.farfield_tolerance)


@pytest.mark.parametrize('config_file', ('examples/medford/Medford_lines.yml',
                                         'examples/medford/Medford_lines.xml'))
def test_pyproj_crs(config_file):
    ls = lsmaker.LinesinkData(config_file)
    crs = pyproj.CRS.from_epsg(26715)
    assert ls.pyproj_crs == crs


def test_medford_from_lss_xml(lsmaker_instance_with_linesinks):
    ls = lsmaker_instance_with_linesinks
    lss_xml_file = '{}.lss.xml'.format(ls.outfile_basename)
    assert os.path.exists(lss_xml_file)
    ls2 = lsmaker.LinesinkData(GFLOW_lss_xml=lss_xml_file)

    assert ls.ComputationalUnits.lower() == ls2.ComputationalUnits.lower()
    assert ls.BasemapUnits.lower() == ls2.BasemapUnits.lower()

    # compare the linesinks
    compare_df_columns = set(ls2.df.columns).intersection(ls.df.columns)
    # the geometries and coordinates won't be exactly the same
    # explicitly compare the coordinates separately
    compare_df_columns = list(compare_df_columns.difference({'geometry',
                                                        'ls_coords',
                                                        'width'
                                                        }))
    df1 = ls.df[compare_df_columns]
    df2 = ls2.df[compare_df_columns]
    pd.testing.assert_frame_equal(df1, df2, check_dtype=False)
    # compare the coordinates
    for dim in 0, 1:  # (x, y)
        x1 = [crd[dim] for line in ls2.df.ls_coords for crd in line]
        x2 = [crd[dim] for line in ls.df.ls_coords for crd in line]
        assert np.allclose(x1, x2)
    assert np.allclose(ls2.df.width.values, ls.df.width.values, rtol=0.01)
    # set the crs and compare
    ls2.set_crs(epsg=26715)
    assert ls2.pyproj_crs == ls.pyproj_crs


def test_diagnostics(lsmaker_instance_with_linesinks):
    ls = lsmaker_instance_with_linesinks
    ls.run_diagnostics()

    
def test_medford_yaml(lsmaker_instance_from_xml):
    """Test that the xml and yaml config files yield equivalent results.
    """
    config_file = 'examples/medford/Medford_lines.yml'
    ls = lsmaker.LinesinkData(config_file)
    ls == lsmaker_instance_from_xml
    
    
def test_nicolet_example(project_root_path):
    """Test that the xml and yaml config files yield equivalent results.
    """
    os.chdir('examples/nicolet')
    results = subprocess.run(['python', 'make_linesinks.py', 'Nicolet_lines.xml'], shell=True)
    j=2
    os.chdir(project_root_path)
    

@pytest.mark.parametrize('routed_area_entry,multiple_nearfields_list,separate_nearfield_entry',
                         [(True, False, True),
                          (True, False, False),
                          pytest.param(False, False, False, marks=pytest.mark.xfail(
                              reason='multiple nearfields require an enclosed routed area')),
                          (True, True, True),
                          pytest.param(False, True, False, marks=pytest.mark.xfail(
                              reason='multiple nearfields specified as a lit require a separate nearfield entry')),                          
                          ])
def test_multiple_nearfields_dict(test_data_path, routed_area_entry, multiple_nearfields_list,
                                  separate_nearfield_entry):
    config_file = test_data_path / 'test_input.yaml'
    cfg = lsmaker.LinesinkData.load_configuration(config_file)
    
    # multiple nearfields require an enclosed routed area
    # (test should fail if routed_area_entry=True)
    if not routed_area_entry:
        del cfg['ModelDomain']['routed_area']
    
    # option to supply nearfields in a list instead of a dictionary
    # in this case, a separate_nearfield_entry is required
    # to specify a global simplification tolerance for all nearfields in the list 
    if multiple_nearfields_list:
        cfg['ModelDomain']['nearfield'] = list(cfg['ModelDomain']['nearfield'].keys())
        
    # multiple nearfields specified with a dictionary
    # should work with our without a separate "nearfield_tolerance" entry
    # in the former case, the keys in the dictionary will override 
    # the "nearfield_tolerance"
    if separate_nearfield_entry:
        cfg['Simplification']['nearfield_tolerance'] = 1000

    ls = lsmaker.LinesinkData(cfg)
    ls.make_linesinks()
    # check that the linesink geometries are simplified
    # from the original geometries as expected
    assert all(ls.df['geometry'].simplify(ls.df['tol'].values) == ls.df['ls_geom'])
    # the actual coordinate written to the GFLOW LSS XML file
    # should match the simplified linesink geometries
    # except for drainage lakes that may have been further edited
    # in conversion from polygon to linesink around the perimeter
    loc = ~ls.df['waterbody'].values
    assert all([LineString(coords).within(g.buffer(100)) 
                for coords, g in zip(ls.df.loc[loc, 'ls_coords'], 
                                     ls.df.loc[loc, 'ls_geom'])])
    
    # the 1000 m tolerance specified in nearfield_tolerance
    # should not have been assigned in the dictionary case
    if not multiple_nearfields_list:
        assert not any(ls.df['tol'].unique() == 1000)
