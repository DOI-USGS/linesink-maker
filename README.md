| **Notice:** This repository is being migrated to https://github.com/usgs/linesink-maker.git in preparation for a USGS software release. |
| --- |

# linesink-maker
Rapid creation of linesink elements for stream network simulation in the groundwater flow modeling program GFLOW (<http://www.haitjema.com/>). Nearfield (highest resultion), mid-field and farfield (least resolution, zero resistance) areas of a desired stream network can be defined by polygon shapefiles. The LinesinkData are then created from NHDPlus hydrography. The number of resulting linesink equations (level of detail) in the various areas of the stream network can be controlled by a distance tolerance parameter specified by geographic area (defined by one of the polygons). Results are written to a linesink string (LSS) XML file that can be imported into the GFLOW GUI (version 2.2 or higher).
 
![Tests](https://github.com/usgs/linesink-maker/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/usgs/linesink-maker/branch/develop/graph/badge.svg)](https://codecov.io/gh/usgs/linesink-maker)
[![PyPI version](https://badge.fury.io/py/linesink-maker.svg)](https://badge.fury.io/py/linesink-maker)



Getting Started
----------------------------------------------- 
See the [linesink-maker documentation](https://usgs.github.io/linesink-maker/index.html)


Installation
-----------------------------------------------
See the [Installation Instructions](https://usgs.github.io/linesink-maker/latest/installation.html)

How to cite
--------------
###### Citation for Linesink-maker

Leaf, A.T., Fienen, M.N. and Reeves, H.W., 2021. SFRmaker and Linesink-maker: Rapid construction of streamflow routing networks from hydrography data, Groundwater xx (x), xx–xx (in revision). [\<waiting on DOI from journal>](https://doi.org/10.5066/P9U2T031)

###### Software/Code citation for Linesink-maker (IP-122356):
Leaf, A.T., 2021, Linesink-maker version 0.1.3: U.S. Geological Survey Software Release, xx Jan. 2021, [https://doi.org/10.5066/P99QSDDX](https://doi.org/10.5066/P99QSDDX)

Disclaimer
----------

This software is preliminary or provisional and is subject to revision. It is
being provided to meet the need for timely best science. The software has not
received final approval by the U.S. Geological Survey (USGS). No warranty,
expressed or implied, is made by the USGS or the U.S. Government as to the
functionality of the software and related material nor shall the fact of release
constitute any such warranty. The software is provided on the condition that
neither the USGS nor the U.S. Government shall be held liable for any damages
resulting from the authorized or unauthorized use of the software.