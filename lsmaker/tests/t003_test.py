"""Tests based on the Medford example."""
import lsmaker
import os
import pytest


@pytest.fixture(scope='module')
def lsmaker_instance_from_xml():
    config_file = 'examples/medford/Medford_lines.xml'
    ls = lsmaker.LinesinkData(config_file)
    return ls


@pytest.fixture(scope='module')
def lsmaker_instance_with_linesinks(lsmaker_instance_from_xml):
    ls = lsmaker_instance_from_xml
    ls.preprocess(save=True)
    ls.make_linesinks(shp='preprocessed/lines.shp')
    return ls


def test_medford(lsmaker_instance_with_linesinks):
    ls = lsmaker_instance_with_linesinks
    assert isinstance(ls, lsmaker.LinesinkData)


def test_diagnostics(lsmaker_instance_with_linesinks):
    ls = lsmaker_instance_with_linesinks
    ls.run_diagnostics()

    
def test_medford_yaml(lsmaker_instance_from_xml):
    """Test that the xml and yaml config files yield equivalent results.
    """
    config_file = 'examples/medford/Medford_lines.yml'
    ls = lsmaker.LinesinkData(config_file)
    ls == lsmaker_instance_from_xml
