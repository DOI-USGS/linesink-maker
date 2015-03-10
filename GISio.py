# suppress annoying pandas openpyxl warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import gdal
import fiona
from shapely.geometry import Point, shape, asLineString, mapping
from shapely.wkt import loads
import pandas as pd
import shutil

def getPRJwkt(epsg):
   """
   from: https://code.google.com/p/pyshp/wiki/CreatePRJfiles

   Grabs a WKT version of an EPSG code
   usage getPRJwkt(4326)

   This makes use of links like http://spatialreference.org/ref/epsg/4326/prettywkt/
   """
   import urllib
   f=urllib.urlopen("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg))
   return (f.read())

def get_proj4(prj):
    """Get proj4 string for a projection file

    Parameters
    ----------
    prj : string
        Shapefile or projection file

    Returns
    -------
    proj4 string (http://trac.osgeo.org/proj/)

    """
    '''
    Using fiona (couldn't figure out how to do this with just a prj file)
    from fiona.crs import to_string
    c = fiona.open(shp).crs
    proj4 = to_string(c)
    '''
    # using osgeo
    from osgeo import osr

    prjfile = prj[:-4] + '.prj' # allows shp or prj to be argued
    prjtext = open(prjfile).read()
    srs = osr.SpatialReference()
    srs.ImportFromESRI([prjtext])
    proj4 = srs.ExportToProj4()
    return proj4

def shp2df(shplist, index=None, clipto=pd.DataFrame(), true_values=None, false_values=None, \
           skip_empty_geom=True):
    '''
    Read shapefile into Pandas dataframe
    ``shplist`` = (string or list) of shapefile name(s)
    ``index`` = (string) column to use as index for dataframe
    ``geometry`` = (True/False) whether or not to read geometric information
    ``clipto`` = (dataframe) limit what is brought in to items in index of clipto (requires index)
    ``true_values`` = (list) same as argument for pandas read_csv
    ``false_values`` = (list) same as argument for pandas read_csv
    from shapefile into dataframe column "geometry"
    '''
    if isinstance(shplist, str):
        shplist = [shplist]

    if len(clipto) > 0 and index:
        clipto_index = np.ravel(clipto.index)
        clip = True
    else:
        clip = False

    df = pd.DataFrame()
    for shp in shplist:
        print "\nreading {}...".format(shp)
        shp_obj = fiona.open(shp, 'r')

        if index is not None:
            # handle capitolization issues with index field name
            fields = shp_obj.schema['properties'].keys()
            index = [f for f in fields if index.lower() == f.lower()][0]

        attributes = []
        # for reading in shapefiles
        if shp_obj.schema['geometry'] != 'None':
            for line in shp_obj:

                props = line['properties']
                # limit what is brought in to items in index of clipto
                if clip:
                    if not props[index] in clipto_index:
                        continue
                props['geometry'] = line.get('geometry', None)
                attributes.append(props)
            print '--> building dataframe... (may take a while for large shapefiles)'
            shp_df = pd.DataFrame(attributes)

            # handle null geometries
            geoms = shp_df.geometry.tolist()
            if geoms.count(None) == 0:
                shp_df['geometry'] = [shape(g) for g in geoms]
            elif skip_empty_geom:
                null_geoms = [i for i, g in enumerate(geoms) if g is None]
                shp_df.drop(null_geoms, axis=0, inplace=True)
                shp_df['geometry'] = [shape(g) for g in shp_df.geometry.tolist()]
            else:
                shp_df['geometry'] = [shape(g) if g is not None else None
                                      for g in geoms]

        # for reading in DBF files (just like shps, but without geometry)
        else:
            for line in shp_obj:

                props = line['properties']
                # limit what is brought in to items in index of clipto
                if clip:
                    if not props[index] in clipto_index:
                        continue
                attributes.append(props)
            print '--> building dataframe... (may take a while for large shapefiles)'
            shp_df = pd.DataFrame(attributes)

        # set the dataframe index from the index column
        if index is not None:
            shp_df.index = shp_df[index].values

        df = df.append(shp_df)

        # convert any t/f columns to numpy boolean data
        if true_values or false_values:
            replace_boolean = {}
            for t in true_values:
                replace_boolean[t] = True
            for f in false_values:
                replace_boolean[f] = False

            # only remap columns that have values to be replaced
            for c in df.columns:
                if len(set(df[c]).intersection(set(true_values))) > 0:
                    df[c] = df[c].map(replace_boolean)
        
    return df
    

def shp_properties(df):
    # convert dtypes in dataframe to 32 bit
    #i = -1
    for i, dtype in enumerate(df.dtypes.tolist()):
        #i += 1
        if dtype == np.dtype('float64') or df.columns[i] == 'geometry':
            continue
        # need to convert integers to 16-bit for shapefile format
        #elif dtype == np.dtype('int64') or dtype == np.dtype('int32'):
        elif 'float' in dtype.name:
            df[df.columns[i]] = df[df.columns[i]].astype('float64')
        elif dtype == np.dtype('int64'):
            df[df.columns[i]] = df[df.columns[i]].astype('int32')
        elif dtype == np.dtype('bool'):
            df[df.columns[i]] = df[df.columns[i]].astype('str')
        # convert all other datatypes (e.g. tuples) to strings
        else:
            df[df.columns[i]] = df[df.columns[i]].astype('str')
    # strip dtypes just down to 'float' or 'int'
    dtypes = [''.join([c for c in d.name if not c.isdigit()]) for d in list(df.dtypes)]
    #dtypes = [d.name for d in list(df.dtypes)]
    # also exchange any 'object' dtype for 'str'
    dtypes = [d.replace('object', 'str') for d in dtypes]
    properties = dict(zip(df.columns, dtypes))
    return properties


def shpfromdf(df, shpname, Xname, Yname, prj):
    '''
    creates point shape file from pandas dataframe
    shp: name of shapefile to write
    Xname: name of column containing Xcoordinates
    Yname: name of column containing Ycoordinates
    '''
    '''
    # convert dtypes in dataframe to 32 bit
    i = -1
    dtypes = list(df.dtypes)
    for dtype in dtypes:
        i += 1
        if dtype == np.dtype('float64'):
            continue
            #df[df.columns[i]] = df[df.columns[i]].astype('float32')
        elif dtype == np.dtype('int64'):
            df[df.columns[i]] = df[df.columns[i]].astype('int32')
    # strip dtypes just down to 'float' or 'int'
    dtypes = [''.join([c for c in d.name if not c.isdigit()]) for d in list(df.dtypes)]
    # also exchange any 'object' dtype for 'str'
    dtypes = [d.replace('object', 'str') for d in dtypes]

    properties = dict(zip(df.columns, dtypes))
    '''
    properties = shp_properties(df)
    schema = {'geometry': 'Point', 'properties': properties}

    with fiona.collection(shpname, "w", "ESRI Shapefile", schema) as output:
        for i in df.index:
            X = df.iloc[i][Xname]
            Y = df.iloc[i][Yname]
            point = Point(X, Y)
            props = dict(zip(df.columns, df.iloc[i]))
            output.write({'properties': props,
                          'geometry': mapping(point)})
    shutil.copyfile(prj, "{}.prj".format(shpname[:-4]))

def csv2points(csv, X='POINT_X', Y='POINT_Y', shpname=None, prj='EPSG:4326', **kwargs):
    '''
    convert csv with point information to shapefile
    ``**kwargs``: keyword arguments to pandas read_csv()
    '''
    if not shpname:
        shpname = csv[:-4] + '.shp'
    df = pd.read_csv(csv, **kwargs)
    df['geometry'] = [Point(p) for p in zip(df[X], df[Y])]
    df2shp(df, shpname, geo_column='geometry', prj=prj)

def xlsx2points(xlsx, sheetname='Sheet1', X='X', Y='Y', shpname=None, prj='EPSG:4326'):
    '''
    convert Excel file with point information to shapefile
    '''
    if not shpname:
        shpname = xlsx.split('.')[0] + '.shp'
    df = pd.read_excel(xlsx, sheetname)
    df['geometry'] = [Point(p) for p in zip(df[X], df[Y])]
    df2shp(df, shpname, geo_column='geometry', prj=prj)


def df2shp(dataframe, shpname, geo_column='geometry', index=False, prj=None, epsg=None, proj4=None, crs=None):
    '''
    Write a DataFrame to a shapefile
    dataframe: dataframe to write to shapefile
    geo_column: optional column containing geometry to write - default is 'geometry'
    index: If true, write out the dataframe index as a column
    --->there are four ways to specify the projection....choose one
    prj: <file>.prj filename (string)
    epsg: EPSG identifier (integer)
    proj4: pyproj style projection string definition
    crs: crs attribute (dictionary) as read by fiona
    '''

    df = dataframe.copy() # make a copy so the supplied dataframe isn't edited

    # reassign geometry column if geo_column is special (e.g. something other than "geometry")
    if geo_column != 'geometry':
        df['geometry'] = df[geo_column]
        df.drop(geo_column, axis=1, inplace=True)

    # include index in shapefile as an attribute field
    if index:
        if df.index.name is None:
            df.index.name = 'index'
        df[df.index.name] = df.index

    # enforce character limit for names! (otherwise fiona marks it zero)
    # somewhat kludgey, but should work for duplicates up to 99
    df.columns = map(str, df.columns) # convert columns to strings in case some are ints
    overtheline = [(i, '{}{}'.format(c[:8],i)) for i, c in enumerate(df.columns) if len(c) > 10]

    newcolumns = list(df.columns)
    for i, c in overtheline:
        newcolumns[i] = c
    df.columns = newcolumns

    properties = shp_properties(df)
    del properties['geometry']

    # sort the dataframe columns (so that properties coincide)
    df = df.sort(axis=1)

    # set projection (or use a prj file, which must be copied after shp is written)
    # alternatively, provide a crs in dictionary form as read using fiona
    # from a shapefile like fiona.open(inshpfile).crs

    if epsg is not None:
        from fiona.crs import from_epsg
        crs = from_epsg(int(epsg))
    elif proj4 is not None:
        from fiona.crs import from_string
        crs = from_string(proj4)
    elif crs is not None:
        pass
    else:
        pass
    Type = df.iloc[0]['geometry'].type
    schema = {'geometry': Type, 'properties': properties}
    length = len(df)

    props = df.drop('geometry', axis=1).to_dict(orient='records')
    mapped = [mapping(g) for g in df.geometry]
    print 'writing {}...'.format(shpname)
    with fiona.collection(shpname, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as output:
        for i in range(length):
            output.write({'properties': props[i],
                          'geometry': mapped[i]})

    if prj is not None:
        """
        if 'epsg' in prj.lower():
            epsg = int(prj.split(':')[1])
            prjstr = getPRJwkt(epsg).replace('\n', '') # get rid of any EOL
            ofp = open("{}.prj".format(shpname[:-4]), 'w')
            ofp.write(prjstr)
            ofp.close()
        """
        try:
            print 'copying {} --> {}...'.format(prj, "{}.prj".format(shpname[:-4]))
            shutil.copyfile(prj, "{}.prj".format(shpname[:-4]))
        except IOError:
            print 'Warning: could not find specified prj file. shp will not be projected.'



def linestring_shpfromdf(df, shpname, IDname, Xname, Yname, Zname, prj, aggregate=None):
    '''
    creates point shape file from pandas dataframe
    shp: name of shapefile to write
    Xname: name of column containing Xcoordinates
    Yname: name of column containing Ycoordinates
    Zname: name of column containing Zcoordinates
    IDname: column with unique integers for each line
    aggregate = dict of column names (keys) and operations (entries)
    '''

    # setup properties for schema
    # if including other properties besides line identifier,
    # aggregate those to single value for line, using supplied aggregate dictionary
    if aggregate:
        cols = [IDname] + aggregate.keys()
        aggregated = df[cols].groupby(IDname).agg(aggregate)
        aggregated[IDname] = aggregated.index
        properties = shp_properties(aggregated)
    # otherwise setup properties to just include line identifier
    else:
        properties = {IDname: 'int'}
        aggregated = pd.DataFrame(df[IDname].astype('int32'))

    schema = {'geometry': '3D LineString', 'properties': properties}
    lines = list(np.unique(df[IDname].astype('int32')))

    with fiona.collection(shpname, "w", "ESRI Shapefile", schema) as output:
        for line in lines:

            lineinfo = df[df[IDname] == line]
            vertices = lineinfo[[Xname, Yname, Zname]].values
            linestring = asLineString(vertices)
            props = dict(zip(aggregated.columns, aggregated.ix[line, :]))
            output.write({'properties': props,
                          'geometry': mapping(linestring)})
    shutil.copyfile(prj, "{}.prj".format(shpname[:-4]))
    
    
def read_raster(rasterfile):
    '''
    reads a GDAL raster into numpy array for plotting
    also returns meshgrid of x and y coordinates of each cell for plotting
    based on code stolen from:
    http://stackoverflow.com/questions/20488765/plot-gdal-raster-using-matplotlib-basemap 
    '''
    try:
        ds = gdal.Open(rasterfile)
    except:
        raise IOError("problem reading raster file {}".format(rasterfile))

    print '\nreading in {} into numpy array...'.format(rasterfile)
    data = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    
    xres = gt[1]
    yres = gt[5]
    
    # get the edge coordinates and add half the resolution 
    # to go to center coordinates
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] + yres * 0.5
    
    del ds

    print 'creating a grid of xy coordinates in the original projection...'
    xy = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
    
    # create a mask for no-data values
    data[data<-1.0e+20] = 0
    
    return data, gt, proj, xy
    
def flatten_3Dshp(shp, outshape=None):
	
	if not outshape:
	    outshape = '{}_2D.shp'.format(shp[:-4])	
	
	df = shp2df(shp, geometry=True)
	
	# somehow this removes 3D formatting
	df['2D'] = df['geometry'].map(lambda x: loads(x.wkt))
	
	# drop the original geometry column
	df = df.drop('geometry', axis=1)
	
	# poop it back out
	df2shp(df, outshape, '2D', shp[:-4]+'.prj')
	

	
	