.. Packaging Scientific Python documentation master file, created by
   sphinx-quickstart on Thu Jun 28 12:35:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

linesink-maker
============================================
version |version|

.. raw:: html

   <hr>

Rapid creation of linesink elements for stream network simulation in the groundwater flow modeling program GFLOW (<http://www.haitjema.com/>). Nearfield (highest resultion), mid-field and farfield (least resolution, zero resistance) areas of a desired stream network can be defined by polygon shapefiles. The LinesinkData are then created from NHDPlus hydrography. The number of resulting linesink equations (level of detail) in the various areas of the stream network can be controlled by a distance tolerance parameter specified by geographic area (defined by one of the polygons). Results are written to a linesink string (LSS) XML file that can be imported into the GFLOW GUI (version 2.2 or higher).

`Go to the GitHub repository <https://github.com/aleaf/linesink-maker>`_

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   Philosophy <philosophy>
   Installation <installation>
   Examples <examples>

.. toctree::
  :maxdepth: 1
  :caption: Usage

   <usage>

.. toctree::
  :maxdepth: 1
  :caption: Reference

   Code Reference <api/index>
   Configuration defaults <config-file-defaults>
   Release History <release-history>
    
.. toctree::
  :maxdepth: 1
  :caption: Developer

  Contributing to linesink-maker <contributing>