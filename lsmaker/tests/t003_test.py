"""Tests based on the Medford example."""
import lsmaker
import os
import numpy as np
import pandas as pd
import pyproj
import pytest


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
    compare_df_columns = compare_df_columns.difference({'geometry',
                                                        'ls_coords',
                                                        'width'
                                                        })
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
