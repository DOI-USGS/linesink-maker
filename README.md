#LinesinkMaker
Python package to construct linesink networks for the analytic element groundwater flow modeling
program GFLOW (<http://www.haitjema.com/>). 
  
* Reads information from NHDPlus v2 (<http://www.horizon-systems.com/nhdplus/NHDPlusV2_home.php/>)
* Writes a GFLOW linesink string (lss) XML import file, and a shapefile of the 
linesink network.

#####Required python packages:  
* lxml
* fiona
* shapely
* pandas

#####Required GFLOW version:  
Requires a version of GFLOW with the linesink import feature, which is not included in the version currently available on the GFLOW website (2.1.2). According to Henk Haitjema, the linesink import feature will be part of a new GFLOW release that will be made available soon, possibly as early as late April 2015. 

#####Installing the package requirements:
An easy way to install these is through a Python distribution such as Anaconda (<https://store.continuum.io/cshop/anaconda/>). Once Anaconda is installed, packages can be added using the **conda** package manager. For example: 
 
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
This fetches the latest version of the package from the **Python Package Index** (<https://pypi.python.org/pypi>).  
Alternatively, for a binary installer obtained from <http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely>:
    
```  
$ pip install Shapely‑1.5.6‑cp27‑none‑win_amd64.whl
```  

#####to install LinesinkMaker:  
Click either *Clone in Desktop* (if you have the GitHub desktop software installed), or *Download ZIP*, navigate to the (extracted) folder that contains **setup.py** and run:  

```
$ python setup.py install
```

#####to import:
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

#####Model domain specification:  
* shapefile of the model nearfield area (where linesinks will be routed and have resistance)  
* shapefile of the model farfield area (where linesinks will be zero-resistance and not routed)

#####All shapefiles must be in a consistent, projected coordinate system with units of feet or meters.  
Automatic reprojection to a specified system will be added to the code at some point, but is not implemented yet.  

#####Input file for LinesinkMaker
The input files, and other input settings such as default resistance and line simplification tolerances, are specified in an XML input file. See the example folders for a template with input instructions.


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
and then navigate to the notebook (*.ipynb* file). Or, the notebook can also be viewed here:  
<http://nbviewer.ipython.org/github/aleaf/LinesinkMaker/blob/master/example_Nicolet/Nicolet.ipynb>

###Importing the linesink string file into GFLOW  
LinesinkMaker outputs a linesink string file of the form <basename>.lss.xml, which can be imported into GFLOW under ```Tools>Import>Line-sink Strings```. It can also be inspected in any text editor (a editor which support XML code highlighting, such as **Notepad++** or **Text Wrangler** is highlly recommended).  
LinesinkMaker also outputs a shapefile representation of the linesink network (<basename>.shp), for visualization in a GIS.