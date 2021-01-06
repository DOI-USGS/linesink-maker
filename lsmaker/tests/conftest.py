import os
from pathlib import Path
import shutil
import pytest


@pytest.fixture(scope="session")
def project_root_path():
    """Root folder for the project (with setup.py)"""
    filepath = os.path.split(os.path.abspath(__file__))[0]
    return Path(os.path.normpath(os.path.join(filepath, '../../')))


@pytest.fixture(scope="session")
def test_data_path(project_root_path):
    """Datasets for the tests.
    """
    return Path(project_root_path, 'lsmaker', 'tests', 'data')

#@pytest.fixture(scope="session", autouse=True)
#def test_output_path():
#    """Datasets for tests."""
#    test_output_path = 'lsmaker/tests/output'
#    reset = True
#    if reset:
#        if os.path.isdir(test_output_path):
#            shutil.rmtree(test_output_path)
#        os.makedirs(test_output_path)
#    return test_output_path


@pytest.fixture(scope="session", autouse=True)
def test_output_path(project_root_path):
    """(Re)make an output folder for the tests
    at the begining of each test session."""
    folder = project_root_path / 'lsmaker/tests/output'
    reset = True
    if reset:
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
    return folder