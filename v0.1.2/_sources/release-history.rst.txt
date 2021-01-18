===============
Release History
===============

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