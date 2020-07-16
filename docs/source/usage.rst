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
downloads. The NHDPlus files are specified in the XML input file under the tag **\<NHDfiles\>**.


Model domain specification
+++++++++++++++++++++++++++++++++++++++  

* **shapefile of the model nearfield area** (where LinesinkData will be routed and have resistance) **(required)**
* **shapefile of the model farfield area** (where LinesinkData will be zero-resistance and not routed) (**optional**; if no farfield shapefile is provided, a buffer is drawn around the provided nearfield. The default for this buffer is 10,000 basemap units. Alternatively, the size of the buffer can be specified in the XML input file under the tag **\<farfield_buffer\>**.
* in addition, a **third shapefile** defining an intermediate area with routed, resistance LinesinkData can be supplied with the **\<routed_area\>** tag. This allows for 3 levels of line simplification, with the most detail limited to the immediate nearfield.


Line simplification
+++++++++++++++++++++++++++++++++++++++
Linesinkmaker uses the line simplification algorithm in the shapely package to reduce the vertices in the NHDPlus GIS flowline coverages so that a reasonable number of LinesinkData are produced. Vertices are removed until the simplified line deviates from the original line by a specified distance tolerance. Tolerances for the model nearfield and farfield areas are specified in the XML input file (**\<nearfield_tolerance\>** and **\<farfield_tolerance\>farfield_tolerance>**). The user may want to adjust these values depending on the desired level of detail for the model, and the constraint of keeping the linesink equations beneath the maxmimum for GFLOW. Reasonable starting values are 100-200 m for the nearfield, and 300-500 m for the farfield.

Other inputs
+++++++++++++++++++++++++++++++++++++++
Other options, such as minimum lake size and minimum stream order to retain in the model farfield, may be specified in the XML input file. See the example XML input files for more details.


Creating the XML Input file for LinesinkMaker
------------------------------------------------
The input files, and other input settings such as default resistance and line simplification tolerances, are specified in an **XML input file**. See the example folders for templates with input instructions (e.g. **Nicolet_lines.xml**). An editor that supports XML code highlighting, such as **Notepad++** or **Text Wrangler** is highly recommended for working with this file. 



Running LinesinkMaker
--------------------------------
LinesinkMaker can be run from the command line by calling the script make_linesinks.py with an XML input file as an argument::

    python make_linesinks.py Medford_lines.xml



Importing the linesink string file into GFLOW  
----------------------------------------------------------------
LinesinkMaker outputs a linesink string file of the form **\<basename>.lss.xml**, which can be imported into GFLOW under `Tools>Import>Line-sink Strings`. It can also be inspected in any text editor. 

Viewing the LinesinkData in a GIS
----------------------------------------------------------------
LinesinkMaker also outputs a shapefile representation of the linesink network (**\<basename>.shp**), for visualization in a GIS. For an example, see **Medford.shp** after running the :ref:`Medford National Forest Unit` example.

