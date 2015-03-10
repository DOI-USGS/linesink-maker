###Preprocessing Readme

####To make Arcpy work with any other python distribution (e.g. Anaconda):
  
  
1) **Make a file called: Desktop10.pth**

In that file, should be the lines:  
C:\ArcGIS\Desktop10.2\arcpy  
C:\ArcGIS\Desktop10.2\bin64  
C:\ArcGIS\Desktop10.2\ArcToolbox\Scripts  

N.B --> That second line may be "bin" for 32 bit or "bin64" for 64 bit

2) **Place this file in the site packages folder**  
For Anaconda, that is most likely:

`C:\Users\aleaf\AppData\Local\Continuum\Anaconda`


###Manual steps for preprocessing LinesinkMaker input in ArcMap:

#####1) "Clip" the NHDPlus flowlines and waterbodies datasets to the model farfield polygon. 
<http://resources.arcgis.com/en/help/main/10.2/index.html#//000800000004000000>

Save the result of the clipping to new shapefiles, which are the same as those specified in the **<preprocessed_files>** section of the **XML input file**.  

#####2) Perform an "Erase analysis", to cut-out the model nearfield from the farfield polygon 
<http://resources.arcgis.com/en/help/main/10.2/index.html#//00080000000m000000>  
(making the farfield polygon a donut with exterior and interior rings). Save this to file specified as **<farfield_multipolygon>** in the **XML input file**.  

#####3) Run "FeatureToPoint" on the NHD waterbodies dataset, 
<http://resources.arcgis.com/en/help/main/10.2/index.html#//00170000003m000000>  
resulting in a shapefile of points for each waterbody.  

#####4) Run "ExtractMultiValuesToPoints" on the waterbody points created in Step 3 
<http://resources.arcgis.com/en/help/main/10.2/index.html#//009z0000002s000000>  
to get an elevation value for each waterbody **from the DEM** for the area. The name for the resulting point shapefile with elevation attributes should be the same as the name for the clipped waterbodies shapefile specified in the **XML input file**, but with the suffix **"_points.shp"** added.