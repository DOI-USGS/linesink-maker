import glob
import os

import pytest


def included_notebooks():
    include = ['examples']  # folders relative to project_root_path
    # explicitly set paths relative to project_root_path
    # (in case test is called from somewhere else)
    filepath = os.path.split(os.path.abspath(__file__))[0]
    project_root_path = os.path.normpath(os.path.join(filepath, '../../'))
    include = [os.path.join(project_root_path, folder) for folder in include]
    files = []
    for folder in include:
        files += glob.glob(os.path.join(folder, '*.ipynb'))
    return sorted(files)


@pytest.fixture(params=included_notebooks(), scope='module')
def notebook(request):
    return request.param


@pytest.fixture(scope='session')
def kernel_name():
    """Pick a Jupyter Notebook kernel from the ones available.
    """
    import jupyter_client
    M = jupyter_client.kernelspec.KernelSpecManager()
    specs = M.find_kernel_specs()

    # try using the first one of these kernels that is found
    try_kernel_names = ['test', 'lsmaker', 'gis']
    for name in try_kernel_names:
        if name in specs:
            return name
    # otherwise use the first kernel listed in specs
    return list(specs.keys())[0]


# even though test runs locally on Windows 10, and on Travis
def test_notebook(notebook, kernel_name, tmpdir, project_root_path):
    # run autotest on each notebook
    notebook = project_root_path / notebook
    path, fname = os.path.split(notebook)

    # save the rendered notebook to the documentation folder
    # so that nbsphinx can render it in the docs
    # the docs get built when the tests are run on travis
    # so successful execution of this test will build the notebooks for the docs
    output_folder = project_root_path / 'docs/source/notebooks'
    print('saving {} to {}'.format(fname, output_folder))

    cmd = ('jupyter ' + 'nbconvert '
           '--ExecutePreprocessor.timeout=600 '
           '--ExecutePreprocessor.kernel_name={} '.format(kernel_name) +
           '--to ' + 'notebook '
           '--execute ' + '{} '.format(notebook) +
           '--output-dir ' + '{} '.format(output_folder) +
           '--output ' + '{}'.format(fname))
    ival = os.system(cmd)
    assert ival == 0, 'could not run {}'.format(os.path.abspath(notebook))
