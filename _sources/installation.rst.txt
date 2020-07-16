============
Installation
============

Required python packages
------------------------
(see below for installation instructions):

* lxml
* fiona
* shapely
* pandas
* pyproj
* requests
* gisutils


GFLOW 2.2 or higher required
------------------------------------------------
Linesinkmaker requires GFLOW 2.2 or higher, which has the capability of importing linesink string (LSS) xml files.


Installing the required Python packages
++++++++++++++++++++++++++++++++++++++++
The best way to install the required packages through `Anaconda <https://www.anaconda.com/products/individual>`_. Once Anaconda is installed, open a command prompt with Anaconda python in the system path (i.e., so that if you run `>python`, you get the Anaconda python). For Windows users, this probably means running the Anaconda command prompt from the start menu. Note that this shortcut can be copied to any folder for convenience. By right-clicking on the shortcut and selecting "properties", you can also change the starting location of the command prompt to the folder it is in by replacing the "Start in" field with ``%cd%``.

Then at the command prompt::

    conda env create -f requirements.yml


This creates a conda environment with the required packages listed in requirements.yml. To use the environment::

    conda activate lsmaker

You should see `(lsmaker)` to the left of the command prompt. 

.. note::
    **The `lsmaker` conda environment needs to be activated to use linksink-maker!**


To install LinesinkMaker
++++++++++++++++++++++++++++++++++++++++++++++++++++  
From the main page of the `GitHub repository <https://github.com/aleaf/linesink-maker>`_
, click on the green Code button in the upper right, and select an option to clone or download the repository. Once the files have downloaded, navigate to the ``linesink-maker`` folder (which should contain **setup.py**) and run::

    python setup.py install


(Windows users can launch a command line at the folder by right clicking on the folder icon and then choosing *Open command window here*)  
