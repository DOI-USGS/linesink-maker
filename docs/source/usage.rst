Input requirements
------------------

.. note::
   All shapefile input should include projection information (``\*.prj`` files), and be in a standard coordinate reference system that is well known (for example, by a valid EPSG identifier). CRS units of feet are not recommended!

NHDPlus files
+++++++++++++  
For each major drainage area encompassed by the model (e.g. Area 04 representing the Great Lakes Basin, 07 representing the Upper Mississippi Basin, etc):  

* NHDFlowline.shp  
* NHDWaterbody.shp  
* elevslope.dbf  
* PlusFlowlineVAA.dbf

These are available at: <http://www.horizon-systems.com/nhdplus/NHDPlusV2_data.php>  in the **`NHDPlusV21_GL_04_NHDSnapshot_07.7z`** and **`NHDPlusV21_GL_04_NHDPlusAttributes_08.7z`** 
downloads. The NHDPlus files are specified in the YAML input file under the ``flowlines:``, ``elevslope:``, ``PlusFlowVAA:`` and ``waterbodies:`` keys, as illustrated in the `Medford National Forest Unit example <https://aleaf.github.io/linesink-maker/medford.html>`_.


Model domain specification
+++++++++++++++++++++++++++++++++++++++  

* **shapefile of the model nearfield area** (where LinesinkData will be routed and have resistance) **(required)**
* **shapefile of the model farfield area** (where LinesinkData will be zero-resistance and not routed) (**optional**; if no farfield shapefile is provided, a buffer is drawn around the provided nearfield. The default for this buffer is 10,000 basemap units. Alternatively, the size of the buffer can be specified in the YAML input file under the ``farfield_buffer:`` key.
* in addition, a **third shapefile** defining an intermediate area with routed, resistance LinesinkData can be supplied with the ``routed_area:`` key. This allows for 3 levels of line simplification, with the most detail limited to the immediate nearfield.


Line simplification
+++++++++++++++++++++++++++++++++++++++
Linesink-maker uses the line simplification algorithm in the shapely package to reduce the vertices in the NHDPlus GIS flowline coverages so that a reasonable number of LinesinkData are produced. Vertices are removed until the simplified line deviates from the original line by a specified distance tolerance. Tolerances for the model nearfield and farfield areas are specified in the YAML input file (``nearfield_tolerance:`` and ``farfield_tolerance:``). The user may want to adjust these values depending on the desired level of detail for the model, and the constraint of keeping the linesink equations beneath the maxmimum for GFLOW. Reasonable starting values are 100-200 m for the nearfield, and 300-500 m for the farfield.

Other inputs
+++++++++++++++++++++++++++++++++++++++
Other options, such as minimum lake size and minimum stream order to retain in the model farfield, may be specified in the YAML input file. 


Creating the YAML Input file for Linesink-maker
------------------------------------------------
The above inputs and other settings are specified in a configuration file using the `YAML format <yaml.org>`_, which maps ``key: value`` pairs similar to a Python dictionary. See the :ref:`Medford National Forest Unit` example for more details for more details on how to make a configuration file. An editor that supports YAML code highlighting, such as `VS Code <https://code.visualstudio.com/>`_, `BBEdit <https://www.barebones.com/products/bbedit/>`_ or `Sublime Text <https://www.sublimetext.com/>`_ is highly recommended. A :ref:`default configuration file <Default configuration settings for Linesink-maker>` contains a more comprehensive listing of settings and default values that are used for any variables not specified by the user.


Running Linesink-maker
--------------------------------
Linesink-maker can be run with the following python script by replacing ``'Medford_lines.yml'`` with the name of your configuration file. 

.. literalinclude:: ../../examples/medford/make_linesinks.py
    :language: python
    :linenos:

Which can be executed at the command line with::

    python make_linesinks.py


Diagnosing errors
--------------------------------
After making the Linesinks, Linesink-maker runs the :meth:`~lsmaker.lsmaker.LinesinkData.run_diagnostics` method, which checks for common issues such as zero gradients (in streambed), duplicate vertices, and linesinks that cross one another (a common side-effect of simplification). Results of the diagnostics are reported in file specified under the ``error_reporting:`` option in the configuration file (which defaults to **linesinkMaker_errors.txt**). Errors are generally referenced by NHDPlus COMID, allowing users to visualize the errors by importing the linesink shapefile into a GIS environment (see below). The errors can then be resolved either in the input linework, or within the GFLOW GUI after importing the lines.

Importing the linesink string file into GFLOW
----------------------------------------------------------------
Linesink-maker outputs a linesink string file named **\<basename>.lss.xml**, where `basename` is the name specified under the ``outfile_basename:`` option in the configuration file (default **'Model'**). The steps for importing the linesinks into GFLOW are summarized below. Additional details are available in the GFLOW documentation.

1) Create a new GFLOW database
2) Make sure the computational and GIS units are properly specified within the GFLOW GUI
3) Then within the GFLOW GUI, from the `Tools` menu, select `Import > Line-sink Strings`.

Viewing the LinesinkData in a GIS
----------------------------------------------------------------
Linesink-maker also outputs a shapefile representation of the linesink network (**\<basename>.shp**), for visualization in a GIS. For an example, see **Medford.shp** after running the :ref:`Medford National Forest Unit` example.
