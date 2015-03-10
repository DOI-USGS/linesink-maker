#LinesinkMaker
Python package to construct linesink networks for the analytic element groundwater flow modeling
program GFLOW (<http://www.haitjema.com/>). 
  
* Reads information from NHDPlus v2 (<http://www.horizon-systems.com/nhdplus/NHDPlusV2_home.php/>)
* Writes a GFLOW linesink string (lss) XML import file, and a shapefile of the 
linesink network.

#####Required python packages (see below for installation instructions):  
* lxml
* fiona
* shapely
* pandas

###! Software dependency caveats:
Unfortunately in addition to the above python modules there are two additional software dependencies for the time being that are somewhat problematic:  

* **Arcpy**: Linesink maker requires some preprocessing steps that are automated by the preprocess_arcpy() method (see the IPython Notebook and/or make_linesinks.py files in the example folders). This dependency will be removed soon in a future version of LinesinkMaker.  
  
	* 	If you have ArcGIS version 10 or later on your machine, you can make Arcpy available to use by following the steps in the preproc_readme.md file.
	* Alternatively, these steps can be done manually in any GIS software. See the preproc_readme.md for instructions.	
  
* **GFLOW Linesink import capability**:  
Output from LinesinkMaker is imported into GFLOW using the linesink import feature, which is not included in the version currently available on the GFLOW website (2.1.2). According to Henk Haitjema, the linesink import feature will be part of a new GFLOW release that will be made available soon, possibly as early as late April 2015. If you already have a GFLOW license and want to try out LinesinkMaker before then, send me a message!

###Installing the required Python packages:
LinesinkMaker runs in Python 2.7. An easy way to install the required packages (and to install and manage Python in general) is through a Python distribution such as **Anaconda**, available for free at (<https://store.continuum.io/cshop/anaconda/>). Once Anaconda is installed, packages can be added at the command line (on any platform) using the **conda** package manager. For example: 
 
```
$ conda install fiona  
```
Windows users may need to install shapely from a binary installer, which can be obtained from <http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely>, or via **pip**. Pip can be installed using **conda**:  

```
$ conda install pip  
```  
then
  
```  
$ pip install shapely  
```  
This fetches the latest version of the **shapely** package from the **Python Package Index** (<https://pypi.python.org/pypi>).  
Alternatively, for a binary installer obtained from <http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely>:
    
```  
$ pip install Shapely‑1.5.6‑cp27‑none‑win_amd64.whl
```  

###to install LinesinkMaker:  
From this page, click either *Clone in Desktop* (if you have the GitHub desktop software installed), or *Download ZIP*, navigate to the (extracted) folder that contains **setup.py** and run:  

```
$ python setup.py install
```

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

These are available at: <http://www.horizon-systems.com/nhdplus/NHDPlusV2_data.php>  in the **NHDPlusV21_GL_04_NHDSnapshot_07.7z** and **NHDPlusV21_GL_04_NHDPlusAttributes_08.7z** downloads.
#####A DEM of the model domain:
A DEM is required to obtain elevations for lakes not connected to the stream network. The DEM is only read by the **preprocess_arcpy()** method, so would not be needed if theses steps are completed manually (**see preproc_readme.md**).


#####Model domain specification:  
* shapefile of the model nearfield area (where linesinks will be routed and have resistance)  
* shapefile of the model farfield area (where linesinks will be zero-resistance and not routed)

#####All shapefiles must be in a consistent, projected coordinate system with units of feet or meters.  
Automatic reprojection to a specified system will be added to the code soon, but is not implemented yet.  

##Creating the XML Input file for LinesinkMaker
The input files, and other input settings such as default resistance and line simplification tolerances, are specified in an **XML input file**. See the example folders for templates with input instructions (e.g. **Nicolet_lines.xml**). An editor that supports XML code highlighting, such as **Notepad++** or **Text Wrangler** is highly recommended for working with this file. 



##Running LinesinkMaker

LinesinkMaker can be run from the command line as a script, or interactively in an environment such as **IPython Notebook** (<http://ipython.org/notebook.html>). The **example_Nicolet** and **example_Medford** folders have examples of both approaches.
  
#####Running from the command line:
First edit the script **make_linesinks.py** so that it points to the correct **XML input file** and then run by calling:  

```
\>python make_linesinks.py
```
#####Running from IPython Notebooks:  
from the **example_Nicolte** folder, or from a parent folder:  

```
\>ipython notebook
```
and then navigate to the notebook (*.ipynb* file). Or, the notebook can be viewed here:  
<http://nbviewer.ipython.org/github/aleaf/LinesinkMaker/blob/master/example_Nicolet/Nicolet.ipynb>

###Importing the linesink string file into GFLOW  
LinesinkMaker outputs a linesink string file of the form **\<basename>.lss.xml**, which can be imported into GFLOW under ```Tools>Import>Line-sink Strings```. It can also be inspected in any text editor.  
###Viewing the linesinks in a GIS
LinesinkMaker also outputs a shapefile representation of the linesink network (**\<basename>.shp**), for visualization in a GIS.