===============
Release History
===============

Version 0.2 (2024-02-20)
--------------------------
* add option to :ref:`specify multiple nearfield areas <Specifying one or more nearfields>`
* fix an issue where lines crossing lakes were getting over-simplified in the process of correcting the crosses.

Version 0.1.6 (2023-11-27)
--------------------------
* fix issue with seepage lake elevations: National Map EPQS URL changed, which was resulting in zero elevations.
* fix issue that occurred when NHDPlus COMIDs were read in as floats

Version 0.1.5 (2023-11-07)
--------------------------
* fix issue with preprocessing routine in lsmaker.py, where `AttributeError: 'dict' object has no attribute 'type'` was occuring on writing farfield and routed area shapefiles.

Version 0.1.4 (2022-08-25)
--------------------------
* fix issue with lsmaker.py that was resulting in ``ValueError: Linestrings must have at least 2 coordinate tuples`` (related to changes to pandas)
* add geopandas as dependency. Could eventually be used for all shapefile reading and writing where feasible, and easy handling of coorindate references and transformations.
* other minor fixes and clean-up

Version 0.1.3 (2021-01-29)
--------------------------
USGS software release associated with `Groundwater` publication

Version 0.1.2 (2021-01-06)
--------------------------
* fix bug in writing of lss.xml output where ComputationalUnits and BasemapUnits were lower case
* some minor refactoring to address warnings and open file point issues on Windows with the log file

Version 0.1.1 (2020-11-04)
--------------------------
* refactor utils module to support `gis-utils <https://github.com/aleaf/gis-utils>`_
* update Exporting and Visualizing GFLOW output example

Version 0.1 (Initial Release; 2020-08-06)
------------------------------------------
* implement YAML as primary config file format (XML still supported)
* adopted a `modified version <https://github.com/aleaf/scientific-python-cookiecutter>`_ of the `Scientific Python Cookiecutter <https://github.com/NSLS-II/scientific-python-cookiecutter>`_ structure
* added documentation, testing, CI, etc.
* add pyproj CRS attribute as more robust way of managing coordinate reference systems
* drop arcpy-related code

Pre-release history
---------------------
* see prior GitHub commits for "prehistory" of the project dating back to the start of the project in 2015