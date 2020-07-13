import os
import shutil
import pytest


@pytest.fixture(scope="session")
def project_root_path():
    filepath = os.path.split(os.path.abspath(__file__))[0]
    return os.path.normpath(os.path.join(filepath, '../../'))


@pytest.fixture(scope="session")
def test_data_path():
    """Datasets for tests."""
    return 'lsmaker/tests/data'


@pytest.fixture(scope="session", autouse=True)
def test_output_path():
    """Datasets for tests."""
    test_output_path = 'lsmaker/tests/output'
    reset = True
    if reset:
        if os.path.isdir(test_output_path):
            shutil.rmtree(test_output_path)
        os.makedirs(test_output_path)
    return test_output_path