"""Tests based on the Medford example."""
import lsmaker
import os
import pytest


@pytest.fixture
def lsmaker_instance_from_xml():
    config_file = 'examples/example_Medford/Medford_lines.xml'
    ls = lsmaker.LinesinkData(config_file)
    return ls


def test_medford(lsmaker_instance_from_xml):

    #path = 'examples/example_Medford/Medford_lines.xml'
    #dir, input_file = os.path.split(path)
#
    #os.chdir(dir)
    #ls = lsmaker.LinesinkData(input_file)
    ls = lsmaker_instance_from_xml

    ls.preprocess(save=True)

    ls.make_linesinks(shp='preprocessed/lines.shp')

    
def test_medford_yaml(lsmaker_instance_from_xml):
    """Test that the xml and yaml config files yield equivalent results.
    """
    config_file = 'examples/example_Medford/Medford_lines.yml'
    ls = lsmaker.LinesinkData(config_file)
    ls == lsmaker_instance_from_xml
    