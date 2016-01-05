#LinesinkMaker
Python package to construct linesink networks for the analytic element groundwater flow modeling
program GFLOW (<http://www.haitjema.com/>). 
  
* Reads information from NHDPlus v2 (<http://www.horizon-systems.com/nhdplus/NHDPlusV2_home.php/>)
* Writes a GFLOW linesink string (lss) XML import file, and a shapefile of the 
linesink network.

#####Required python packages (see below for installation instructions):  
* lxml
* urllib2
* json
* fiona
* shapely
* pandas
* pyproj

  
###GFLOW Linesink import capability
Linesinkmaker requires GFLOW 2.2 or higher, which has the capability of importing linesink string xml files.

###Installing the required Python packages:
LinesinkMaker runs in Python 2.7. An easy way to install the required packages (and to install and manage Python in general) is through a Python distribution such as **Anaconda**, available for free at (<https://store.continuum.io/cshop/anaconda/>). Once Anaconda is installed, packages can be added at the command line (on any platform) using the **conda** package manager. For example: 
 
```
$ conda install fiona  
```
Fiona, shapely, and pyproj depend on compiled GIS libraries. Instructions for installing these packages with their underlying libraries are given here: <https://github.com/aleaf/SFRmaker/blob/master/pythonGIS_install_readme.md>

###to install LinesinkMaker:  
From this page, click either *Clone in Desktop* (if you have the GitHub desktop software installed), or *Download ZIP*. Once the files have downloaded, navigate to the LinesinkMaker (which should contain **setup.py**) and run:  

```
$ python setup.py install
```  
(Windows users can launch a command line at the folder by right clicking on the folder icon and then choosing *Open command window here*)  


#####to import into a python session:
```
import lsmaker
```



##Required input  
#####From NHDPlus v2:  
For each major drainage area encompassed by the model (e.g. Area 04 representing the Great Lakes Basin, 07 representing the Upper Mississippi Basin, etc):  

* NHDFlowline.shp  
* NHDWaterbody.shp  
* elevslope.dbf  
* PlusFlowlineVAA.dbf

These are available at: <http://www.horizon-systems.com/nhdplus/NHDPlusV2_data.php>  in the **NHDPlusV21_GL_04_NHDSnapshot_07.7z** and **NHDPlusV21_GL_04_NHDPlusAttributes_08.7z** 
downloads. The NHDPlus files are specified in the XML input file under the tag **\<NHDfiles\>**.

#####Model domain specification:  
* shapefile of the model nearfield area (where linesinks will be routed and have resistance)  
   **(required)**
* shapefile of the model farfield area (where linesinks will be zero-resistance and not routed)  
 (**optional**; if no farfield shapefile is provided, a buffer is drawn around the provided nearfield. The default for this buffer is 10,000 basemap units. Alternatively, the size of the buffer can be specified in the XML input file under the tag **\<farfield_buffer\>**.
 
#####Line simplification
Linesinkmaker uses the line simplification algorithm in the shapely package to reduce the vertices in the NHDPlus GIS flowline coverages so that a reasonable number of linesinks are produced. Vertices are removed until the simplified line deviates from the original line by a specified distance tolerance. Tolerances for the model nearfield and farfield areas are specified in the XML input file (**\<nearfield_tolerance\>** and **\<farfield_tolerance\>farfield_tolerance>**). The user may want to adjust these values depending on the desired level of detail for the model, and the constraint of keeping the linesink equations beneath the maxmimum for GFLOW. Reasonable starting values are 100-200 m for the nearfield, and 300-500 m for the farfield.

#####Other inputs
Other options, such as minimum lake size and minimum stream order to retain in the model farfield, may be specified in the XML input file. See the example XML input files for more details.


##Creating the XML Input file for LinesinkMaker
The input files, and other input settings such as default resistance and line simplification tolerances, are specified in an **XML input file**. See the example folders for templates with input instructions (e.g. **Nicolet_lines.xml**). An editor that supports XML code highlighting, such as **Notepad++** or **Text Wrangler** is highly recommended for working with this file. 



##Running LinesinkMaker

LinesinkMaker can be run from the command line by calling the script make_linesinks.py with an XML input file as an argument:

```
\>python make_linesinks.py Medford_lines.xml
```


###Importing the linesink string file into GFLOW  
LinesinkMaker outputs a linesink string file of the form **\<basename>.lss.xml**, which can be imported into GFLOW under ```Tools>Import>Line-sink Strings```. It can also be inspected in any text editor.  
###Viewing the linesinks in a GIS
LinesinkMaker also outputs a shapefile representation of the linesink network (**\<basename>.shp**), for visualization in a GIS. For an example, see **Medford.shp** after running the Medford example.

###Uninstall
```
pip uninstall lsmaker
```