# linesink-maker
Rapid creation of linesink elements for stream network simulation in the groundwater flow modeling program GFLOW (<http://www.haitjema.com/>). Nearfield (highest resultion), mid-field and farfield (least resolution, zero resistance) areas of a desired stream network can be defined by polygon shapefiles. The LinesinkData are then created from NHDPlus hydrography. The number of resulting linesink equations (level of detail) in the various areas of the stream network can be controlled by a distance tolerance parameter specified by geographic area (defined by one of the polygons). Results are written to a linesink string (LSS) XML file that can be imported into the GFLOW GUI (version 2.2 or higher).
 
[![Build Status](https://img.shields.io/travis/aleaf/linesink-maker.svg)](https://travis-ci.com/aleaf/linesink-maker) [![codecov](https://codecov.io/gh/aleaf/linesink-maker/branch/develop/graph/badge.svg)](https://codecov.io/gh/aleaf/linesink-maker)



Getting Started
----------------------------------------------- 
See the [linesink-maker documentation](https://aleaf.github.io/linesink-maker/index.html)


Installation
-----------------------------------------------
See the [Installation Instructions](https://aleaf.github.io/linesink-maker/installation.html)
