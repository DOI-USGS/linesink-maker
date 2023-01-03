from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from lsmaker.lsmaker import *
from lsmaker.diagnostics import Diagnostics
from .utils import *

from . import _version
__version__ = _version.get_versions()['version']
