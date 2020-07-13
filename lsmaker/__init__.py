from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

#from . import utils
from lsmaker.lsmaker import *
from lsmaker.diagnostics import Diagnostics
#from . import GISio
