__author__ = 'aleaf'

import xml.etree.ElementTree as ET
import sys
import numpy as np
import os
import pandas as pd
import shutil

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import json
from functools import partial
import fiona
from fiona.crs import to_string
from shapely.geometry import Polygon, LineString, Point, shape, mapping
from shapely.ops import unary_union, transform
import pyproj
import math
try:
    import matplotlib.pyplot as plt
except:
    pass
from .diagnostics import *

# ## Functions #############################
def add_projection(line, point):
    """Add vertex to line at point,
    if the closest point on the line isn't an end.

    Parameters
    ----------
    line : LineString
    point : Point

    Returns
    -------
    newline : LineString
        Line with point added, or original line, if point coincides with end.
    """
    l = line
    mp = point
    distance = l.project(mp)
    if distance <= 0.0 or distance >= line.length:
        return line
    coords = list(l.coords)
    for i, p in enumerate(l.coords):
        pd = l.project(Point(p))
        if pd == distance:
            return line
        elif pd > distance:
            return LineString(coords[:i] + [(mp.x, mp.y)] + coords[i:])

def add_vertices_at_testpoints(lssdf, tpgeoms, tol=200):
    """Add vertices to linesinks at locations of testpoints
    (so that modeled flow observations are correct)

    Parameters
    ----------
    lssdf : DataFrame
        DataFrame of linesink strings. Must contain 'geometry' column
        of shapely LineStrings defining geometries of linesink strings,
        in same coordinate system as testpoints.

    tpgeoms : list
        List of testpoint geometries.

    tol : float
        Tolerance, in coordinate system units, for considering lines
        near the testpoints.

    Returns
    -------
    geoms : list of geometries
        New geometry column with added vertices.

    """
    df = lssdf.copy()

    for mp in tpgeoms:

        # find all lines within tolerance
        nearby = np.array([l.intersects(mp.buffer(tol)) for l in df.geometry])
        ldf = df[nearby].copy()

        # choose closest if two or more nearby lines
        if len(ldf) > 1:
            ldf['dist'] = [mp.distance(ll.interpolate(ll.project(mp)))
                           for ll in ldf.geometry.values]
            ldf.sort_values('dist', inplace=True)

        # if at least one line is nearby
        if len(ldf) > 0:
            ind = ldf.index[0]
            l = ldf.geometry.values[0]
            newline = add_projection(l, mp)
            df.set_value(ind, 'geometry', newline)
    return df.geometry.tolist()

def get_elevations_from_epqs(points, units='Feet'):
    """From list of shapely points in lat, lon, returns list of elevation values
    """
    print('querying National Map Elevation Point Query Service...')
    try:
        elevations = [get_elevation_from_epqs(p.x, p.y, units=units) for p in points]
    except:
        e = sys.exc_info()[0]
        print(e)
        print('Problem accessing Elevation Point Query Service. '
              'Need an internet connection to get seepage lake elevations.')
        quit()
    return elevations


def get_elevation_from_epqs(lon, lat, units='Feet'):
    """Returns an elevation value at a point location.

    Notes
    -----
    example url for -91, 45:
        http://nationalmap.gov/epqs/pqs.php?x=-91&y=45&units=Feet&output=json

    Examples
    --------
    >>> get_elevation_from_epqs(-91, 45)
    1139.778277
    """
    url = 'http://nationalmap.gov/epqs/pqs.php?'
    url += 'x={}&y={}&units={}&output=json'.format(lon, lat, units)
    epqsdata = urlopen(url).readline()
    elev = json.loads(epqsdata.decode())['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']
    #json.loads(epqsdata)
    if not isinstance(elev, float):
        print(('Warning, invalid elevation of {} returned for {}, {}.\nSetting elevation to 0.'.format(elev, lon, lat)))
        elev = 0.0
    return elev


def _get_random_point_in_polygon(poly):
    """Generates a point within a polygon (for lakes where the centroid is not in the lake)"""
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if poly.contains(p):
            return p


def reproject(geoms, pr1, pr2):
    """Reprojects a list of geometries from coordinate system pr1 to pr2
    (given as proj4 strings)."""
    print('reprojecting from {} to {}...'.format(pr1, pr2))
    if not isinstance(geoms, list):
        geoms = [geoms]
    pr1 = pyproj.Proj(pr1, errcheck=True, preserve_units=True)
    pr2 = pyproj.Proj(pr2, errcheck=True, preserve_units=True)
    project = partial(pyproj.transform, pr1, pr2)
    return [transform(project, g) for g in geoms]


def w_parameter(B, lmbda):
    """Compute w parameter for estimating an effective conductance term
    (i.e., when simulating Lakes using Linesinks instead of GFLOW's lake package)

    If only larger lakes are simulated (e.g., > 1 km2), w parameter will be = lambda

    see Haitjema 2005, "Dealing with Resistance to Flow into Surface Waters"
    """
    if lmbda <= 0.1 * B:
        w = lmbda
    elif 0.1 * B < lmbda < 2 * B:
        w = lmbda * np.tanh(B / (2 * lmbda))
    else:
        w = B / 2
    return w


def width_from_arboate(arbolate, lmbda):
    """Estimate stream width in feet from arbolate sum in meters, using relationship
    described by Feinstein et al (2010), Appendix 2, p 266.
    """
    estwidth = 0.1193 * math.pow(1000 * arbolate, 0.5032)
    w = 2 * w_parameter(estwidth, lmbda)  # assumes stream is rep. by single linesink
    return w


def lake_width(area, total_line_length, lmbda):
    """Estimate conductance width from lake area and length of flowlines running through it
    """
    if total_line_length > 0:
        estwidth = 1000 * (area / total_line_length) / 0.3048  # (km2/km)*(ft/km)
    else:
        estwidth = np.sqrt(area / np.pi) * 1000 / 0.3048  # (km)*(ft/km)

    # see Haitjema 2005, "Dealing with Resistance to Flow into Surface Waters"
    # basically if only larger lakes are simulated (e.g., > 1 km2), w parameter will be = lambda
    # this assumes that GFLOW's lake package will not be used
    w = w_parameter(estwidth, lmbda)
    return w  # feet


def name(x):
    """Abbreviations for naming linesinks from names in NHDPlus
    GFLOW requires linesink names to be 32 characters or less
    """
    if x.GNIS_NAME:
        # reduce name down with abbreviations
        abb = {'Branch': 'Br',
               'Creek': 'Crk',
               'East': 'E',
               'Flowage': 'Fl',
               'Lake': 'L',
               'North': 'N',
               'Pond': 'P',
               'Reservoir': 'Res',
               'River': 'R',
               'South': 'S',
               'West': 'W'}

        name = '{} {}'.format(x.name, x.GNIS_NAME)
        for k, v in abb.items():
            name = name.replace(k, v)
    else:
        name = '{} unnamed'.format(x.name)
    return name[:32]


def uniquelist(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def closest_vertex_ind(point, shape_coords):
    """Returns index of closest vertices in shapely geometry object
    Ugly but works
    """
    crds = shape_coords
    X = np.array([i[0] for i in crds])
    Y = np.array([i[1] for i in crds])
    dX, dY = X - point[0], Y - point[1]
    closest_ind = np.argmin(np.sqrt(dX ** 2 + dY ** 2))
    return closest_ind


def move_point_along_line(x1, x2, dist):
    diff = (x2[0] - x1[0], x2[1] - x1[1])
    return tuple(x2 - dist * np.sign(diff))


class linesinks:
    maxlines = 4000

    int_dtype = np.int64

    dtypes = {'Label': str,
                    'HeadSpecified': float,
                    'StartingHead': float,
                    'EndingHead': float,
                    'Resistance': float,
                    'Width': float,
                    'Depth': float,
                    'Routing': np.int64,
                    'EndStream': np.int64,
                    'OverlandFlow': np.int64,
                    'EndInflow': np.int64,
                    'ScenResistance': str,
                    'Drain': np.int64,
                    'ScenFluxName': str,
                    'Gallery': np.int64,
                    'TotalDischarge': np.int64,
                    'InletStream': np.int64,
                    'OutletStream': np.int64,
                    'OutletTable': str,
                    'Lake': np.int64,
                    'Precipitation': float,
                    'Evapotranspiration': float,
                    'Farfield': bool,
                    'chkScenario': bool,
                    'AutoSWIZC': np.int64,
                    'DefaultResistance': float}

    fcodes = {'Perennial': 46006,
              'Intermittent': 46003,
              'Uncategorized': 46000}

    def __init__(self, infile):

        try:
            inpardat = ET.parse(infile)
        except:
            raise InputFileMissing

        inpars = inpardat.getroot()
        self.inpars = inpars

        # setup the working directory (default to current directory)
        try:
            self.path = inpars.findall('.//working_dir')[0].text
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        except:
            self.path = os.getcwd()

        # global settings
        self.preproc = self.tf2flag(self._get_XMLentry('preproc', 'True'))
        self.resistance = self._get_XMLentry('resistance', 0.3, float)  # (days); c in documentation
        self.H = self._get_XMLentry('H', 100, float) # aquifer thickness in model units
        self.k = self._get_XMLentry('k', 10, float)  # hydraulic conductivity of the aquifer in model units
        self.lmbda = np.sqrt(self.k * self.H * self.resistance)
        self.ScenResistance = self._get_XMLentry('ScenResistance', 'linesink')
        self.global_streambed_thickness = self._get_XMLentry('global_streambed_thickness',
                                                             3, float)  # streambed thickness
        self.ComputationalUnits = self._get_XMLentry('ComputationalUnits', 'Feet') # 'Feet' or 'Meters'; for XML output file
        self.BasemapUnits = self._get_XMLentry('BasemapUnits', 'Meters')
        # elevation units multiplier (from NHDPlus cm to model units)
        self.z_mult = 0.03280839895013123 if self.ComputationalUnits.lower() == 'feet' else 0.01

        # model domain
        self.farfield = self._get_XMLentry('farfield', None)
        self.routed_area = self._get_XMLentry('routed_area', None)
        self.nearfield = self._get_XMLentry('nearfield', None)
        self.prj = self._get_XMLentry('prj', self.nearfield[:-4] +'.prj')
        self.farfield_buffer = self._get_XMLentry('farfield_buffer', 10000, int)
        self.clip_farfield = self.tf2flag(self._get_XMLentry('clip_farfield', 'False'))
        self.split_by_HUC = self.tf2flag(self._get_XMLentry('split_by_HUC', 'False'))
        self.HUC_shp = self._get_XMLentry('HUC_shp', None)
        self.HUC_name_field = self.tf2flag(self._get_XMLentry('HUC_name_field', 'False'))
        self.crs = fiona.open(self.nearfield).crs # fiona-style coordinate system (proj4 string as dictionary)
        self.crs_str = to_string(self.crs) # self.crs, in proj4 string format

        # simplification
        self.refinement_areas = []  # list of n areas within routed area with additional refinement
        self.nearfield_tolerance = self._get_XMLentry('nearfield_tolerance', 100, int)
        self.routed_area_tolerance = self._get_XMLentry('routed_area_tolerance', 100, int)
        self.farfield_tolerance = self._get_XMLentry('farfield_tolerance', 300, int)
        self.min_farfield_order = self._get_XMLentry('min_farfield_order', 2, int)
        self.min_nearfield_wb_size = self._get_XMLentry('min_nearfield_waterbody_size', 1.0, float)
        self.min_waterbody_size = self._get_XMLentry('min_waterbody_size', 1.0, float)
        self.min_farfield_wb_size = self._get_XMLentry('min_farfield_waterbody_size',
                                                       self.min_waterbody_size, float)
        self.drop_intermittent = self.tf2flag(self._get_XMLentry('drop_intermittent', 'False'))
        self.drop_crossing = self.tf2flag(self._get_XMLentry('drop_crossing', 'False'))

        # NHD files
        self.flowlines = [f.text for f in inpars.findall('.//flowlines')]
        self.elevslope = [f.text for f in inpars.findall('.//elevslope')]
        self.PlusFlowVAA = [f.text for f in inpars.findall('.//PlusFlowVAA')]
        self.waterbodies = [f.text for f in inpars.findall('.//waterbodies')]
        # columns to retain in NHD files (when joining to GIS lines)
        # Note: may need to add method to handle case discrepancies
        self.flowlines_cols = ['COMID', 'FCODE', 'FDATE', 'FLOWDIR', 'FTYPE', 'GNIS_ID', 'GNIS_NAME', 'LENGTHKM',
                               'REACHCODE', 'RESOLUTION', 'WBAREACOMI', 'geometry']
        self.flowlines_cols_dtypes = {'COMID': self.int_dtype,
                                      'FCODE': self.int_dtype,
                                      'FDATE': str,
                                      'FLOWDIR': str,
                                      'FTYPE': str,
                                      'GNIS_ID': self.int_dtype,
                                      'GNIS_NAME': str,
                                      'LENGTHKM': float,
                                      'REACHCODE': str,
                                      'RESOLUTION': str,
                                      'WBAREACOMI': self.int_dtype,
                                      'geometry': object}
        self.elevslope_cols = ['MINELEVSMO', 'MAXELEVSMO']
        self.elevslope_dtypes = {'MINELEVSMO': float,
                                 'MAXELEVSMO': float}
        self.pfvaa_cols = ['ArbolateSu', 'Hydroseq', 'DnHydroseq', 'StreamOrde']
        self.pfvaa_cols_dtypes = {'ArbolateSu': float,
                                  'Hydroseq': self.int_dtype,
                                  'DnHydroseq': self.int_dtype,
                                  'StreamOrde': np.int64}
        self.wb_cols = ['AREASQKM', 'COMID', 'ELEVATION', 'FCODE', 'FDATE', 'FTYPE', 'GNIS_ID', 'GNIS_NAME',
                        'REACHCODE', 'RESOLUTION', 'geometry']
        self.wb_cols_dtypes = {'AREASQKM': float,
                               'COMID': self.int_dtype,
                               'ELEVATION': float,
                               'FCODE': self.int_dtype,
                               'FDATE': str,
                               'FTYPE': str,
                               'GNIS_ID': self.int_dtype,
                               'GNIS_NAME': str,
                               'REACHCODE': str,
                               'RESOLUTION': str,
                               'geometry': object}
        # could do away with above and have one dtypes list
        self.dtypes.update(self.flowlines_cols_dtypes)
        self.dtypes.update(self.elevslope_dtypes)
        self.dtypes.update(self.pfvaa_cols_dtypes)
        self.dtypes.update(self.wb_cols_dtypes)

        # preprocessed files
        self.DEM = self._get_XMLentry('DEM', None)
        self.elevs_field = self._get_XMLentry('elevs_field', 'elev')
        self.DEM_zmult = self._get_XMLentry('DEM_zmult', 1.0, float)

        self.flowlines_clipped = self._get_XMLentry('flowlines_clipped', 'preprocessed/flowlines_clipped.shp')
        self.waterbodies_clipped = self._get_XMLentry('waterbodies_clipped', 'preprocessed/waterbodies_clipped.shp')
        self.routed_mp = self._get_XMLentry('routed_area_multipolygon', 'preprocessed/ra_cutout.shp')
        self.farfield_mp = self._get_XMLentry('farfield_multipolygon', 'preprocessed/ff_cutout.shp')
        self.preprocessed_lines = self._get_XMLentry('preprocessed_lines', 'preprocessed/lines.shp')
        self.preprocdir = os.path.split(self.flowlines_clipped)[0]

        self.wb_centroids_w_elevations = self.waterbodies_clipped[
                                         :-4] + '_points.shp'  # elevations extracted during preprocessing routine
        self.elevs_field = 'DEM'  # field in wb_centroids_w_elevations containing elevations

        # outputs
        self.outfile_basename = inpars.findall('.//outfile_basename')[0].text
        self.error_reporting = inpars.findall('.//error_reporting')[0].text
        self.efp = open(self.error_reporting, 'w')

        # attributes
        self.df = pd.DataFrame() # working dataframe for translating NHDPlus data to linesink strings
        self.lss = pd.DataFrame() # dataframe of GFLOW linesink strings (with attributes)
        self.outsegs = pd.DataFrame()
        self.confluences = pd.DataFrame()

    def _get_XMLentry(self, XMLentry, default_name, dtype=str):
        try:
            txt = self.inpars.findall('.//{}'.format(XMLentry))[0].text
            return dtype(txt) if txt is not None else None
        except:
            return default_name

    def _enforce_dtypes(self, df):
        """Ensure that dataframe column dtypes are correct."""
        for c in df.columns:
            dtype = self.dtypes.get(c, df[c].dtype)
            try:
                df[c] = df[c].astype(dtype)
            except ValueError:
                # handle Nonetype values in some NHDPlus fields
                continue

    def load_poly(self, shapefile, dest_crs=None):
        """Load a shapefile and return the first polygon from its records,
        projected in the destination coordinate system."""
        if dest_crs is None:
            dest_crs = self.crs_str
        print('reading {}...'.format(shapefile))
        shp = fiona.open(shapefile)
        geom = shape(shp.next()['geometry'])
        shp_crs_str = to_string(shp.crs)
        if shp_crs_str != dest_crs:
            geom = reproject(geom, shp_crs_str, dest_crs)[0]
        return geom

    def tf2flag(self, intxt):
        # converts text written in XML file to True or False flag
        if intxt.lower() == 'true':
            return True
        else:
            return False

    def preprocess_arcpy(self):
        """
        requires arcpy

        This method performs the following steps:

        1) clip the NHDPlus flowlines and waterbodies datasets to the model farfield polygon.
           Save the result of the clipping to new shapefiles, which are the same as those specified
           in the <preprocessed_files> section of the XML input file.
        2) Perform an "Erase analysis", to cut-out the model nearfield from the farfield polygon
           (making the farfield polygon a donut with exterior and interior rings). Save this
           to file specified as <farfield_multipolygon> in the XML input file.
        3) Run "FeatureToPoint" on the NHD waterbodies dataset, resulting in a shapefile of points
           for each waterbody.
        4) Run "ExtractMultiValuesToPoints" on the waterbody points created in Step 3 and the DEM
           for the area, to get an elevation value for each waterbody. The name for the resulting
           point shapefile with elevation attributes should be the same as the name for the clipped
           waterbodies shapefile specified in the XML input file, but with the suffix "_points.shp"
           added.

        Notes:
        -----
        **This method does not perform any projections or transformations.** Therefore all input shapefiles must
        be in the same projected coordinate system that will be used in the GFLOW GUI.

        Alternatively, these steps can be performed manually prior to running the preprocess() method.
        """
        try:
            import arcpy
        except:
            print('Could not import arcpy, which is required for this method. Alternatively, these steps can' \
                  'be performed manually prior to running the preprocess() method:\n' \
                  '1) clip the NHDPlus flowlines and waterbodies datasets to the model farfield polygon. \n' \
                  '   Save the result of the clipping to new shapefiles, which are the same as those specified\n' \
                  '   in the <preprocessed_files> section of the XML input file.\n' \
                  '2) Perform an "Erase analysis", to cut-out the model nearfield from the farfield polygon\n' \
                  '   (making the farfield polygon a donut with exterior and interior rings). Save this' \
                  '   to file specified as <farfield_multipolygon> in the XML input file.\n' \
                  '3) Run "FeatureToPoint" on the NHD waterbodies dataset, resulting in a shapefile of points\n' \
                  '   for each waterbody.' \
                  '4) Run "ExtractMultiValuesToPoints" on the waterbody points created in Step 3 and the DEM\n' \
                  '   for the area, to get an elevation value for each waterbody. The name for the resulting\n ' \
                  '   point shapefile with elevation attributes should be the same as the name for the clipped\n' \
                  '   waterbodies shapefile specified in the XML input file, but with the suffix "_points.shp"\n' \
                  '   added.')
        path = self.preprocdir
        flowlines_clipped = os.path.split(self.flowlines_clipped)[1]
        waterbodies_clipped = os.path.split(self.waterbodies_clipped)[1]
        farfield_mp = os.path.split(self.farfield_mp)[1]
        wb_centroids_w_elevations = os.path.split(self.wb_centroids_w_elevations)[1]

        # make the output directory if it doesn't exist yet
        if len(self.preprocdir) > 0 and not os.path.isdir(self.preprocdir):
            os.makedirs(self.preprocdir)

        # initialize the arcpy environment
        arcpy.env.workspace = path
        arcpy.env.overwriteOutput = True
        arcpy.env.qualifiedFieldNames = False
        arcpy.CheckOutExtension("spatial")  # Check spatial analyst license

        if len(self.flowlines) > 1:
            arcpy.Merge_management(self.flowlines, os.path.join(path, 'flowlines_merged.shp'))
            self.flowlines = os.path.join(path, 'flowlines_merged.shp')
        else:
            self.flowlines = self.flowlines[0]
        if len(self.waterbodies) > 1:
            arcpy.Merge_management(self.waterbodies, os.path.join(path, 'waterbodies_merged.shp'))
            self.waterbodies = os.path.join(path, 'waterbodies_merged.shp')
        else:
            self.waterbodies = self.waterbodies[0]

        # make projection file that is independent of any shapefile
        arcpy.Delete_management('GFLOW.prj')
        shutil.copy(self.prj, 'GFLOW.prj')
        self.prj = 'GFLOW.prj'

        print('clipping and reprojecting input datasets...')
        for attr in ['nearfield', 'farfield']:
            shp = self.__dict__[attr]
            if open(self.prj).readline() != open(shp[:-4] + '.prj').readline():
                arcpy.Project_management(shp, 'preprocessed/' + os.path.split(shp)[1], self.prj)
                self.__dict__[attr] = 'preprocessed/' + os.path.split(shp)[1]
                print('reprojected {} to coordinate system in {}...'.format(self.__dict__[attr], self.prj))

        for indata, output in list({self.flowlines: flowlines_clipped, self.waterbodies: waterbodies_clipped}.items()):

            print('clipping {} to extent of {}...'.format(indata, self.farfield))
            arcpy.Clip_analysis(indata, self.farfield, 'tmp.shp')

            if open(self.prj).readline() != open(indata[:-4] + '.prj').readline():
                print('\nreprojecting {} to coordinate system in {}...'.format(indata, self.prj))
                arcpy.Project_management('tmp.shp', output, self.prj)
            else:
                arcpy.Rename_management('tmp.shp', output)
            arcpy.Delete_management('tmp.shp')

        print('\nremoving interior from farfield polygon...')
        arcpy.Erase_analysis(self.farfield, self.nearfield, farfield_mp)
        print('farfield donut written to {}'.format(farfield_mp))

        print('\ngetting NHD Waterbody elevations from DEM (needed for isolated lakes)')
        arcpy.FeatureToPoint_management(waterbodies_clipped, wb_centroids_w_elevations)
        arcpy.sa.ExtractMultiValuesToPoints(wb_centroids_w_elevations, [[self.DEM, self.elevs_field]])
        print('waterbody elevations written to point dataset {}'.format(wb_centroids_w_elevations))
        print('\nDone.')

    def preprocess_python(self):
        """temporary method to take the place of preprocess_arcpy.
        Will eventually combine with preprocess to reduce I/O
        This method also removes the need for a DEM, by getting lake elevations from the National Map Elevation Point Query Service
        """

        # make the output directory if it doesn't exist yet
        if len(self.preprocdir) > 0 and not os.path.isdir(self.preprocdir):
            os.makedirs(self.preprocdir)

        # make projection file that is independent of any shapefile
        if os.path.exists('GFLOW.prj'):
            os.remove('GFLOW.prj')
        if self.prj is not None and os.path.exists(self.prj):
            shutil.copy(self.prj, 'GFLOW.prj')
        self.prj = 'GFLOW.prj'

        def _get_random_point_in_polygon(poly):
            (minx, miny, maxx, maxy) = poly.bounds
            while True:
                p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
                if poly.contains(p):
                    return p

        print('clipping and reprojecting input datasets...')

        self.nf = self.load_poly(self.nearfield)
        self.ra = self.load_poly(self.routed_area)
        self.ff = self.load_poly(self.farfield)

        for attr, shapefiles in list({'fl': self.flowlines, 'wb': self.waterbodies}.items()):
            # all flowlines and waterbodies must be in same coordinate system
            # sfrmaker preproc also uses crs from first file in list
            shp_crs_str = to_string(fiona.open(shapefiles[0]).crs)
            self.__dict__[attr] = GISio.shp2df(shapefiles)
            # if NHD features not in model area coodinate system, reproject
            if shp_crs_str != self.crs_str:
                self.__dict__[attr]['geometry'] = reproject(self.__dict__[attr].geometry.tolist(), shp_crs_str, self.crs_str)
            # 1816107
            # for now, take intersection (don't truncate flowlines at farfield boundary)
            print('clipping to {}...'.format(self.farfield))
            intersects = np.array([l.intersects(self.ff) for l in self.__dict__[attr].geometry])
            self.__dict__[attr] = self.__dict__[attr][intersects].copy()
        if self.clip_farfield:
            print('truncating waterbodies at farfield boundary...')
            # get rid of any islands in the process
            self.wb['geometry'] = [Polygon(g.exterior).intersection(self.ff) for g in self.wb.geometry]
            # for multipolygons, retain largest part
            geoms = self.wb.geometry.values
            for i, g in enumerate(geoms):
                if g.type == 'MultiPolygon':
                    geoms[i] = np.array(g.geoms)[np.argsort([p.area for p in g.geoms])][-1]
            self.wb['geometry'] = geoms

        # for now, write out preprocessed data to maintain structure with arcpy
        GISio.df2shp(self.fl, self.flowlines_clipped, proj4=self.crs_str)
        GISio.df2shp(self.wb, self.waterbodies_clipped, proj4=self.crs_str)

        print('\nmaking donut polygon of farfield (with routed area removed)...')
        ffdonut = self.ff.difference(self.ra)
        with fiona.drivers():
            with fiona.open(self.routed_area) as src:
                with fiona.open(self.farfield_mp, 'w', **src.meta) as output:
                    print(('writing {}'.format(self.farfield_mp)))
                    f = next(src)
                    f['geometry'] = mapping(ffdonut)
                    output.write(f)

        if self.routed_area is not None and self.nearfield is not None:
            print('\nmaking donut polygon of routed area (with nearfield area removed)...')
            if not self.nf.within(self.ra):
                raise ValueError('Nearfield area must be within routed area!')

            donut = self.ra.difference(self.nf)
            with fiona.drivers():
                with fiona.open(self.nearfield) as src:
                    with fiona.open(self.routed_mp, 'w', **src.meta) as output:
                        print(('writing {}'.format(self.routed_mp)))
                        f = next(src)
                        f['geometry'] = mapping(donut)
                        output.write(f)

        print('\ngetting elevations for waterbodies not in the stream network')
        # get elevations for all waterbodies for the time being.
        # otherwise waterbodies associated with first-order streams may be dropped from farfield,
        # but still waterbodies list, causing a key error in the lakes setup
        # these lakes probably should be left in the farfield unless they are explicitly not wanted
        isolated_wb = self.wb
        isolated_wb_comids = isolated_wb.COMID.tolist()
        # (the centroids of some lakes might not be in the lake itself)
        wb_points = [wb.centroid if wb.centroid.within(wb)
                     else _get_random_point_in_polygon(wb)
                     for wb in isolated_wb.geometry.tolist()]
        # reproject the representative points into lat lon
        wb_points_ll = reproject(wb_points, self.crs_str, "+init=epsg:4269")
        wb_elevations = get_elevations_from_epqs(wb_points_ll, units=self.ComputationalUnits)

        # add in elevations from waterbodies in stream network

        wb_points_df = pd.DataFrame({'COMID': isolated_wb_comids,
                                     self.elevs_field: wb_elevations,
                                     'geometry': wb_points})
        GISio.df2shp(wb_points_df, self.wb_centroids_w_elevations, proj4=self.crs_str)

    def preprocess(self, save=True, use_arcpy=False):
        """
        This method associates attribute information in the NHDPlus PlusFlowVAA and Elevslope tables, and
        the model domain configuration (nearfield, farfield, and any other polygon areas) with the NHDPlus
        Flowlines and Waterbodies datasets. The following edits are made to the Flowlines and waterbodies:
        * remove farfield streams lower than <min_farfield_order>
        * remove waterbodies that aren't lakes, and lakes smaller than <min_waterbody_size>
        * convert lakes from polygons to lines; merge the lakes with the with flowlines

        Parameters:
        -----------
        save: True/False
            Saves the preprocessed dataset to a shapefile specified by <preprocessed_lines> in the XML input file

        """

        if self.routed_area is None:
            self.routed_area = self.nearfield
        if self.nearfield is None and self.routed_area is None:
            raise InputFileMissing('Need to supply shapefile of routed area or nearfield.')

        if self.farfield is None:
            print(('\nNo farfield shapefile supplied.\n'
                  'Creating farfield using buffer of {:.1f} {} around model nearfield.\n'
                  .format(self.farfield_buffer, self.BasemapUnits)))
            modelareafile = fiona.open(self.routed_area)
            nfarea = shape(modelareafile[0]['geometry'])
            modelarea_farfield = nfarea.buffer(self.farfield_buffer)
            self.farfield = self.nearfield[:-4] + '_ff.shp'
            output = fiona.open(self.farfield, 'w',
                                crs=modelareafile.crs,
                                schema=modelareafile.schema,
                                driver=modelareafile.driver)
            output.write({'properties': modelareafile[0]['properties'],
                          'geometry': mapping(modelarea_farfield)})
            output.close()


        if use_arcpy:
            self.preprocess_arcpy()
        else:
            self.preprocess_python()

        # open error reporting file
        self.efp = open(self.error_reporting, 'a')
        self.efp.write('\nPreprocessing...\n')

        print('\nAssembling input...')
        # read linework shapefile into pandas dataframe
        df = GISio.shp2df(self.flowlines_clipped, index='COMID', index_dtype=self.int_dtype).drop_duplicates('COMID')
        df.drop([c for c in df.columns if c.lower() not in [cc.lower() for cc in self.flowlines_cols]],
                axis=1, inplace=True)
        # might want to consider enforcing integer index here if strings cause problems
        clipto = df.index.tolist()
        elevs = GISio.shp2df(self.elevslope, index='COMID', index_dtype=self.int_dtype, clipto=clipto)
        pfvaa = GISio.shp2df(self.PlusFlowVAA, index='COMID', index_dtype=self.int_dtype, clipto=clipto)
        wbs = GISio.shp2df(self.waterbodies_clipped, index='COMID', index_dtype=self.int_dtype).drop_duplicates('COMID')
        wbs.drop([c for c in wbs.columns if c.lower() not in [cc.lower() for cc in self.wb_cols]],
                 axis=1, inplace=True)
        self._enforce_dtypes(wbs)

        # check for MultiLineStrings / MultiPolygons and drop them (these are features that were fragmented by the boundaries)
        mls = [i for i in df.index if 'Multi' in df.ix[i, 'geometry'].type]
        df = df.drop(mls, axis=0)
        mps = [i for i in wbs.index if 'Multi' in wbs.ix[i, 'geometry'].type]
        wbs = wbs.drop(mps, axis=0)

        # join NHD tables to lines
        df = df.join(elevs[self.elevslope_cols], how='inner', lsuffix='1')
        df = df.join(pfvaa[self.pfvaa_cols], how='inner', lsuffix='1')
        self._enforce_dtypes(df)

        # read in nearfield and farfield boundaries
        nf = GISio.shp2df(self.nearfield)
        nfg = nf.iloc[0]['geometry']  # polygon representing nearfield
        if self.routed_area != self.nearfield:
            ra = GISio.shp2df(self.routed_mp)
            rag = ra.iloc[0]['geometry']
        else:
            rag = nfg
            nfg = Polygon() # no nearfield specified
        ff = GISio.shp2df(self.farfield_mp)
        ffg = ff.iloc[0]['geometry']  # shapely geometry object for farfield (polygon with interior ring for nearfield)

        print('\nidentifying farfield and nearfield linesinks...')
        df['farfield'] = [line.intersects(ffg) and not line.intersects(rag) for line in df.geometry]
        wbs['farfield'] = [poly.intersects(ffg) for poly in wbs.geometry]
        df['routed'] = [line.intersects(rag) for line in df.geometry]
        wbs['routed'] = [poly.intersects(rag) for poly in wbs.geometry]
        df['nearfield'] = [line.intersects(nfg) for line in df.geometry]
        wbs['nearfield'] = [poly.intersects(nfg) for poly in wbs.geometry]

        print('removing farfield streams lower than {} order...'.format(self.min_farfield_order))
        # retain all streams in the nearfield or in the farfield and of order > min_farfield_order
        df = df[~df.farfield.values | (df.farfield.values & (df.StreamOrde.values >= self.min_farfield_order))]

        farfield_retain = ~(df.farfield.values & (df.FCODE == self.fcodes['Intermittent']).values) # drop intermittent streams from farfield
        if self.drop_intermittent:
            print('removing intermittent streams from routed area outside of nearfield...')
            # retain all streams in the nearfield or in the farfield and of order > min_farfield_order
            retain = farfield_retain & ~(df.routed.values & ~df.nearfield.values & (df.FCODE == self.fcodes['Intermittent']).values)
        df = df[retain].copy()

        print('dropping waterbodies from routed area that are not lakes larger than {}...'.format(self.min_waterbody_size))
        nearfield_wbs = wbs.nearfield.values & (wbs.AREASQKM > self.min_nearfield_wb_size) & (wbs.FTYPE == 'LakePond')
        routedarea_wbs = wbs.routed.values & (wbs.AREASQKM > self.min_waterbody_size) & (wbs.FTYPE == 'LakePond')
        farfield_wbs = wbs.farfield.values & (wbs.AREASQKM > self.min_farfield_wb_size) & (wbs.FTYPE == 'LakePond')

        print('dropping waterbodies from nearfield that are not lakes larger than {}...\n'
              'dropping waterbodies from farfield that are not lakes larger than {}...'.format(self.min_waterbody_size,
                                                                                               self.min_farfield_wb_size))
        wbs = wbs[nearfield_wbs | routedarea_wbs | farfield_wbs]

        print('merging waterbodies with coincident boundaries...')
        dropped = []
        for wb_comid in wbs.index:

            # skipped waterbodies that have already been merged
            if wb_comid in dropped:
                continue

            wb_geometry = wbs.geometry[wb_comid]
            overlapping = wbs.ix[[wb_geometry.intersects(r) for r in wbs.geometry]]
            basering_comid = overlapping.sort_values('FTYPE').index[0]  # sort to prioritize features with names

            # two or more shapes in overlapping signifies a coincident boundary
            if len(overlapping > 1):
                merged = unary_union([r for r in overlapping.geometry])
                # multipolygons will result if the two polygons only have a single point in common
                if merged.type == 'MultiPolygon':
                    continue

                wbs.loc[basering_comid, 'geometry'] = merged

                todrop = [wbc for wbc in overlapping.index if wbc != basering_comid]
                dropped += todrop
                wbs = wbs.drop(todrop, axis=0)  # only keep merged feature; drop others from index

                # replace references to dropped waterbody in lines
                df.loc[df.WBAREACOMI.isin(todrop), 'WBAREACOMI'] = basering_comid

        # swap out polygons in lake geometry column with the linear rings that make up their exteriors
        print('converting lake exterior polygons to lines...')
        wbs['geometry'] = [LineString(g.exterior) for g in wbs.geometry]
        wbs['waterbody'] = [True] * len(wbs)  # boolean variable indicate whether feature is waterbody

        print('merging flowline and waterbody datasets...')
        df['waterbody'] = [False] * len(df)
        df = df.append(wbs)
        df.COMID = df.index
        '''
        if self.clip_farfield:
            print('clipping farfield features...')
            df.loc[df.farfield.values, 'geometry'] = [g.intersection(ffg)
                                                      for g in df.geometry[df.farfield.values]]
        '''
        print('\nDone with preprocessing.')
        if save:
            GISio.df2shp(df, self.preprocessed_lines, proj4=self.crs_str)

        self.df = df

    def simplify_lines(self, nearfield_tolerance=None, routed_area_tolerance=None, farfield_tolerance=None,
                       nearfield_refinement={}):
        """Reduces the number of vertices in the GIS linework representing streams and lakes,
        to within specified tolerances. The tolerance values represent the maximum distance
        in the coordinate system units that the simplified feature can deviate from the original feature.

        Parameters:
        ----------
        nearfield_tolerance : float
            Tolerance for the area representing the model nearfield
        farfield_tolerance : float
            Tolerance for the area representing the model farfield

        Returns:
        -------
        df : DataFrame
            A copy of the df attribute with a 'ls_geom' column of simplified geometries, and
            a 'ls_coords' column containing lists of coordinate tuples defining each simplified line.
        """
        if not hasattr(self, 'df'):
            print('No dataframe attribute for linesinks instance. Run preprocess first.')
            return

        if nearfield_tolerance is None:
            nearfield_tolerance = self.nearfield_tolerance
            routed_area_tolerance = self.routed_area_tolerance
            farfield_tolerance = self.farfield_tolerance

        #if isinstance(self.df.farfield.iloc[0], str):
        #    self.df.loc[:, 'farfield'] = [True if f.lower() == 'true' else False for f in self.df.farfield]

        print('simplifying NHD linework geometries...')
        # simplify line and waterbody geometries
        #(see http://toblerity.org/shapely/manual.html)
        df = self.df[['farfield', 'routed', 'nearfield', 'geometry']].copy()

        ls_geom = np.array([LineString()] * len(df))
        tols = {'nf': nearfield_tolerance,
                'ra': routed_area_tolerance,
                'ff': farfield_tolerance}
        simplification = {'nf': df.nearfield.values,
                          'ra': df.routed.values & ~df.nearfield.values & ~df.farfield.values,
                          'ff': df.farfield.values}

        for k, within in simplification.items():
            # simplify the linesinks in the domain; add simplified geometries to global geometry column
            # assign geometries to numpy array first and then to df (had trouble assigning with pandas)
            ls_geom[within] = [g.simplify(tols[k]) for g in df.ix[within, 'geometry'].tolist()]

        df['ls_geom'] = ls_geom

        # add columns for additional nearfield refinement areas
        for shp in list(nearfield_refinement.keys()):
            if shp not in self.refinement_areas:
                self.refinement_areas.append(shp)
            area_name = os.path.split(shp)[-1][:-4]
            poly = shape(fiona.open(shp).next()['geometry'])
            df[area_name] = [g.intersects(poly) for g in df.geometry]

        # add column of lists, containing linesink coordinates
        df['ls_coords'] = [list(g.coords) for g in df.ls_geom]

        return df

    def prototype(self, nftol=[10, 50, 100, 200, 500], fftol=500):
        """Function to compare multiple simplification distance tolerance values for the model nearfield.

        Parameters:
        -----------
        nftol : list
            Contains the tolerance values to be compared.
        fftol : numeric
            Single tolerance value to be used for the farfield in all comparisons.

        Returns:
        --------
        A new directory called "prototypes" is made

        """
        if not os.path.isdir('prototypes'):
            os.makedirs('prototypes')

        if isinstance(fftol, float) or isinstance(fftol, int):
            fftol = [fftol] * len(nftol)

        nlines = []
        for i, tol in enumerate(nftol):
            df = self.simplify_lines(nearfield_tolerance=tol, farfield_tolerance=fftol[i])

            # count the number of lines with distance tolerance
            nlines.append(np.sum([len(l) for l in df.ls_coords]))

            # make a shapefile of the simplified lines with nearfield_tol=tol
            df.drop(['ls_coords', 'geometry'], axis=1, inplace=True)
            outshp = 'prototypes/' + self.outfile_basename + '_dis_tol_{}.shp'.format(tol)
            GISio.df2shp(df, outshp, geo_column='ls_geom', proj4=self.crs_str)

        plt.figure()
        plt.plot(nftol, nlines)
        plt.xlabel('Distance tolerance')
        plt.ylabel('Number of lines')
        plt.savefig(self.outfile_basename + 'tol_vs_nlines.pdf')

    def adjust_zero_gradient(self, df, increment=0.01):

        dg = Diagnostics(lsm_object=self)
        dg.df = df
        comids0 = dg.check4zero_gradient()

        if len(comids0) > 0:

            if len(self.confluences) == 0:
                self.df = df
                self.map_confluences()
            if len(self.outsegs) == 0:
                self.df = df
                self.map_outsegs()

            self.efp.write('\nzero-gradient adjustments:\n')
            self.efp.write('comid, old_elevmax, old_elevmin, new_elevmax, new_elevmin, downcomid\n')

            print("adjusting elevations for comids with zero-gradient...")
            for comid in comids0:

                outsegs = [o for o in self.outsegs.ix[comid].values if o > 0]
                for i, o in enumerate(outsegs):

                    if i == len(outsegs) - 1:
                        oo = 0
                    else:
                        oo = outsegs[i + 1]

                    minElev, maxElev = df.ix[o, 'minElev'], df.ix[o, 'maxElev']
                    # test if current segment has flat or negative gradient
                    if minElev >= maxElev:
                        minElev = maxElev - increment
                        df.loc[o, 'minElev'] = minElev
                        self.efp.write('{}, {:.2f}, {:.2f}, {:.2f}, {}\n'.format(o, maxElev,
                                                                                 minElev + increment,
                                                                                 maxElev,
                                                                                 minElev, oo))
                        # test if next segment is now higher
                        if oo > 0 and df.ix[oo, 'maxElev'] > minElev:
                            self.efp.write('{}, {:.2f}, {:.2f}, {:.2f}, {}\n'.format(outsegs[i + 1],
                                                                                     df.ix[outsegs[i + 1], 'maxElev'],
                                                                                     df.ix[outsegs[i + 1], 'minElev'],
                                                                                     minElev,
                                                                                     df.ix[outsegs[i + 1], 'minElev'],
                                                                                     oo))
                            df.loc[oo, 'maxElev'] = minElev
                    else:
                        break

                        # check again for zero-gradient lines
            dg.df = df
            comids0 = dg.check4zero_gradient()

            if len(comids0) > 0:
                for c in comids0:
                    self.efp.write('{} '.format(c))
                print("\nWarning!, the following comids had zero gradients:\n{}".format(comids0))
                print("routing for these was turned off. Elevations must be fixed manually.\n" \
                      "See {}".format(self.error_reporting))
        return df

    def drop_crossing_lines(self, df):

        dg = Diagnostics(lsm_object=self)
        crosses = dg.check4crossing_lines()

    def drop_duplicates(self, df):
        # loops or braids in NHD linework can result in duplicate lines after simplification
        # create column of line coordinates converted to strings
        df['ls_coords_str'] = [''.join(map(str, coords)) for coords in df.ls_coords]

        # identify duplicates; make common set of up and down comids for duplicates
        duplicates = np.unique(df.ix[df.duplicated('ls_coords_str'), 'ls_coords_str'])
        for dup in duplicates:
            alld = df[df.ls_coords_str == dup]
            upcomids = []
            dncomid = []
            for i, r in alld.iterrows():
                upcomids += r.upcomids
                dncomid += r.dncomid

            upcomids, dncomid = list(set(upcomids)), list(set(dncomid))

            keep_comid = alld.index[0]
            df.set_value(keep_comid, 'upcomids', upcomids)
            df.set_value(keep_comid, 'dncomid', dncomid)
            for u in upcomids:
                df.set_value(u, 'dncomid', [keep_comid])
            for d in dncomid:
                upids = set(df.ix[d, 'upcomids']).difference(set(alld.index[1:]))
                upids.add(alld.index[0])
                df.set_value(d, 'upcomids', list(upids))
            df.drop(alld.index[1:], axis=0, inplace=True)

        # drop the duplicates (this may cause problems if multiple braids are routed to)
        #df = df.drop_duplicates('ls_coords_str') # drop rows from dataframe containing duplicates
        df = df.drop('ls_coords_str', axis=1)
        return df

    def setup_linesink_lakes(self, df):

        # read in elevations for NHD waterbodies (from preprocessing routine; needed for isolated lakes)
        wb_elevs = GISio.shp2df(self.wb_centroids_w_elevations,
                                index='COMID', index_dtype=self.int_dtype).drop_duplicates('COMID')
        self._enforce_dtypes(wb_elevs)
        wb_elevs = wb_elevs[self.elevs_field] * self.DEM_zmult

        # identify lines that represent lakes
        # get elevations, up/downcomids, and total lengths for those lines
        # assign attributes to lakes, then drop the lines

        df['total_line_length'] = 0  # field to store total shoreline length of lakes
        for wb_comid in self.wblist:

            lines = df[df['WBAREACOMI'] == wb_comid]
            upcomids = []
            dncomids = []

            # isolated lakes have no overlapping lines and no routing
            if len(lines) == 0:
                df.ix[wb_comid, 'maxElev'] = wb_elevs[wb_comid]
                df.ix[wb_comid, 'minElev'] = wb_elevs[wb_comid] - 0.01
                df.ix[wb_comid, 'routing'] = 0
            else:
                df.ix[wb_comid, 'minElev'] = np.min(lines.minElev)
                df.ix[wb_comid, 'maxElev'] = np.min(lines.maxElev)

                # get upcomids and downcomid for lake,
                # by differencing all up/down comids for lines in lake, and comids in the lake
                upcomids = list(set([c for l in lines.upcomids for c in l]) - set(lines.index))
                dncomids = list(set([c for l in lines.dncomid for c in l]) - set(lines.index))

                df.set_value(wb_comid, 'upcomids', upcomids)
                df.set_value(wb_comid, 'dncomid', dncomids)

                # make the lake the down-comid for the upcomids of the lake
                # (instead of the lines that represented the lake in the flowlines dataset)
                # do the same for the down-comid of the lake
                for u in [u for u in upcomids if u > 0]:  # exclude outlets
                    df.set_value(u, 'dncomid', [wb_comid])
                for d in [d for d in dncomids if d > 0]:
                    df.set_value(d, 'upcomids', [wb_comid])
                '''
                # update all up/dn comids in lines dataframe that reference the lines inside of the lakes
                # (replace those references with the comids for the lakes)
                for comid in lines.index:
                    if comid == 937070193:
                        j=2

                    # make the lake the down-comid for the upcomids of the lake
                    # (instead of the lines that represented the lake in the flowlines dataset)
                    df.loc[upcomids, 'dncomid'] = [wb_comid]
                    df.loc[dncomids, 'upcomids'] = [wb_comid]
                    df.ix[df.FTYPE != 'LakePond', 'dncomid'] = [[wb_comid if v == comid else v for v in l] for l in df[df.FTYPE != 'LakePond'].dncomid]
                    df.ix[df.FTYPE != 'LakePond', 'upcomids'] = [[wb_comid if v == comid else v for v in l] for l in df[df.FTYPE != 'LakePond'].upcomids]
                '''
                # get total length of lines representing lake (used later to estimate width)
                df.loc[wb_comid, 'total_line_length'] = np.sum(lines.LENGTHKM)

                # modifications to routed lakes
                #if df.ix[wb_comid, 'routing'] == 1:

                # enforce gradient in routed lakes; update elevations in downstream comids
                if df.ix[wb_comid, 'minElev'] == df.ix[wb_comid, 'maxElev']:
                    df.loc[wb_comid, 'minElev'] -= 0.01
                    dnids = df.ix[wb_comid, 'dncomid']
                    for dnid in [d for d in dnids if d > 0]:
                        df.loc[dnid, 'maxElev'] -= 0.01

            #df['dncomid'] = [[d] if not isinstance(d, list) else d for d in df.dncomid]
            #df['upcomids'] = [[u] if not isinstance(u, list) else u for u in df.upcomids]
            # move begining/end coordinate of linear ring representing lake to outlet location (to ensure correct routing)
            # some routed lakes may not have an outlet
            # do this for both routed and unrouted (farfield) lakes, so that the outlet line won't cross the lake
            # (only tributaries are tested for crossing in step below)
            lake_coords = uniquelist(df.ix[wb_comid, 'ls_coords'])
            if len(df.ix[wb_comid, 'dncomid']) > 0 and dncomids[0] != 0:
                outlet_coords = df.ix[df.ix[wb_comid, 'dncomid'][0], 'ls_coords'][0]
                closest_ind = closest_vertex_ind(outlet_coords, lake_coords)
                lake_coords[closest_ind] = outlet_coords
                next_ind = closest_ind + 1 if closest_ind < (len(lake_coords) - 1) else 0
            # for lakes without outlets, make the last coordinate the outlet so that it can be moved below
            else:
                outlet_coords = lake_coords[-1]
                next_ind = 0

            inlet_coords = move_point_along_line(lake_coords[next_ind], outlet_coords, 1)

            new_coords = [inlet_coords] + lake_coords[next_ind:] + lake_coords[:next_ind]
            df.set_value(wb_comid, 'ls_coords', new_coords)

            # make sure inlets/outlets don't cross lines representing lake
            wb_geom = LineString(df.ix[wb_comid, 'ls_coords'])
            x = [c for c in upcomids if c != 0 and LineString(df.ix[c, 'ls_coords']).crosses(wb_geom)]
            if len(x) > 0:
                for c in x:
                    ls_coords = list(df.ix[c, 'ls_coords'])  # want to copy, to avoid modifying df
                    # find the first intersection point with the lake
                    # (for some reason, two very similar coordinates will be occasionally be returned by intersection)
                    intersection = LineString(ls_coords).intersection(wb_geom)
                    if intersection.type == 'MultiPoint':
                        intersection_point = np.array([intersection.geoms[0].xy[0][0],
                                                       intersection.geoms[0].xy[1][0]])
                    else:
                        intersection_point = np.array([intersection.xy[0][0], intersection.xy[1][0]])
                    # sequentially drop last vertex from line until it no longer crosses the lake
                    crossing = True
                    while crossing:
                        ls_coords.pop(-1)
                        if len(ls_coords) < 2:
                            break
                        # need to test for intersection separately,
                        # in case len(ls_coords) == 1 (can't make a LineString)
                        elif LineString(ls_coords).crosses(wb_geom):
                            break
                    # append new end vertex on line that is close to, but not coincident with lake
                    diff = np.array(ls_coords[-1]) - intersection_point
                    # make a new endpoint that is between the intersection and next to last
                    new_endvert = tuple(intersection_point + np.sign(diff) * np.sqrt(self.nearfield_tolerance))
                    ls_coords.append(new_endvert)
                    df.set_value(c, 'ls_coords', ls_coords)
            # drop the lines representing the lake from the lines dataframe
            df.drop(lines.index, axis=0, inplace=True)
        return df

    def list_updown_comids(self, df):

        farfield = df.COMID[df.farfield].tolist()
        # record up and downstream comids for lines
        lines = [l for l in df.index if l not in self.wblist and l not in farfield]
        #df['dncomid'] = len(df)*[[]]
        #df['upcomids'] = len(df)*[[]]
        #df.ix[lines, 'dncomid'] = [list(df[df['Hydroseq'] == df.ix[i, 'DnHydroseq']].index) for i in lines]
        #df.ix[lines, 'upcomids'] = [list(df[df['DnHydroseq'] == df.ix[i, 'Hydroseq']].index) for i in lines]
        df['upcomids'] = [[]] * len(df)
        df['dncomid'] = [[]] * len(df)
        dncomid, upcomids = [], []
        for l in lines:
            # set up/down comids that are not in the model domain to zero
            dncomid.append([d if d in lines else 0 for d in
                            list(df[df['Hydroseq'] == df.ix[l, 'DnHydroseq']].index)])
            upcomids.append([u if u in lines else 0 for u in
                             list(df[df['DnHydroseq'] == df.ix[l, 'Hydroseq']].index)])

        df.loc[lines, 'upcomids'] = upcomids
        df.loc[lines, 'dncomid'] = dncomid
        # enforce list datatype (apparently pandas flattens lists of lists len()=1 on assignment
        df['dncomid'] = [[d] if not isinstance(d, list) else d for d in df.dncomid]
        return df

    def makeLineSinks(self, shp=None):
        self.efp = open(self.error_reporting, 'a')
        self.efp.write('\nMaking the lines...\n')

        if shp is None:
            shp = self.preprocessed_lines

        try:
            self.df = GISio.shp2df(shp, index='COMID',
                                   index_dtype=self.int_dtype,
                                   true_values=['True'],
                                   false_values=['False'])
            self._enforce_dtypes(self.df)
        except:
            self.preprocess(save=True)

        # enforce integers columns
        self.df.index = self.df.index.astype(self.int_dtype)
        self.df['COMID'] = self.df.COMID.astype(self.int_dtype)

        df = self.df

        # simplify the lines in the df (dataframe) attribute
        self.lines_df = self.simplify_lines()

        # add linesink geometries back in to dataframe
        #df['ls_geom'] = self.lines_df['ls_geom']
        df['ls_coords'] = self.lines_df['ls_coords']

        self.wblist = set(df.ix[df.waterbody].index.values.astype(self.int_dtype)).difference({0})

        print('Assigning attributes for GFLOW input...')

        # routing
        df['routing'] = df['routed'].astype(int)
        #df['routing'] = len(df) * [1]
        #df.loc[df['farfield'], 'routing'] = 0  # turn off all routing in farfield (conversely, nearfield is all routed)

        # linesink elevations (lakes won't be populated yet)
        min_elev_col = [c for c in df.columns if 'minelev' in c.lower()][0]
        max_elev_col = [c for c in df.columns if 'maxelev' in c.lower()][0]
        df['minElev'] = df[min_elev_col] * self.z_mult
        df['maxElev'] = df[max_elev_col] * self.z_mult
        df['dStage'] = df['maxElev'] - df['minElev']

        # list upstream and downstream comids
        df = self.list_updown_comids(df)

        # discard duplicate linesinks that result from braids in NHD and line simplification
        df = self.drop_duplicates(df)

        # method to represent lakes with linesinks
        df = self.setup_linesink_lakes(df)

        print('\nmerging or splitting lines with only two vertices...')
        # find all routed comids with only 1 line; merge with neighboring comids
        # (GFLOW requires two lines for routed streams)

        def bisect(coords):
            # add vertex to middle of single line segment
            coords = np.array(coords)
            mid = 0.5 * (coords[0] + coords[-1])
            new_coords = list(map(tuple, [coords[0], mid, coords[-1]]))
            return new_coords

        df['nlines'] = [len(coords) - 1 for i, coords in enumerate(df.ls_coords)]

        # bisect lines that have only one segment, and are routed
        ls_coords = df.ls_coords.tolist()
        singlesegment = ((df['nlines'] < 2) & (df['routing'] == 1)).values
        df['ls_coords'] = [bisect(line) if singlesegment[i]
                           else line for i, line in enumerate(ls_coords)]

        # fix linesinks where max and min elevations are the same
        df = self.adjust_zero_gradient(df)

        # end streams
        # evaluate whether downstream segment is in farfield
        downstream_ff = []
        for i in range(len(df)):
            try:
                dff = df.ix[df.iloc[i].dncomid[0], 'farfield'].item()
            except:
                dff = True
            downstream_ff.append(dff)

        # set segments with downstream segment in farfield as End Segments
        df['end_stream'] = len(df) * [0]
        df.loc[downstream_ff, 'end_stream'] = 1  # set

        # widths for lines
        arbolate_sum_col = [c for c in df.columns if 'arbolate' in c.lower()][0]
        df['width'] = df[arbolate_sum_col].map(lambda x: width_from_arboate(x, self.lmbda))

        # widths for lakes
        if np.any(df['FTYPE'] == 'LakePond'):
            df.ix[df['FTYPE'] == 'LakePond', 'width'] = \
                np.vectorize(lake_width)(df.ix[df['FTYPE'] == 'LakePond', 'AREASQKM'],
                                         df.ix[df['FTYPE'] == 'LakePond', 'total_line_length'], self.lmbda)

        # resistance
        df['resistance'] = self.resistance
        df.loc[df['farfield'], 'resistance'] = 0

        # depth
        df['depth'] = self.global_streambed_thickness

        # resistance parameter (scenario)
        df['ScenResistance'] = self.ScenResistance
        df.loc[df['farfield'], 'ScenResistance'] = '__NONE__'

        # linesink location
        df.ix[df['FTYPE'] != 'LakePond', 'AutoSWIZC'] = 1  # Along stream centerline
        df.ix[df['FTYPE'] == 'LakePond', 'AutoSWIZC'] = 2  # Along surface water boundary

        # additional check to drop isolated lines
        isolated = [c for c in df.index if len(df.ix[c].dncomid) == 0 and len(df.ix[c].upcomids) == 0
                    and c not in self.wblist]
        #df = df.drop(isolated, axis=0)

        # names
        df['ls_name'] = len(df) * [None]
        df['ls_name'] = df.apply(name, axis=1)

        # compare number of line segments before and after
        npoints_orig = sum([len(p) - 1 for p in df['geometry'].map(lambda x: x.xy[0])])
        npoints_simp = sum([len(p) - 1 for p in df.ls_coords])

        print('\nnumber of lines in original NHD linework: {}'.format(npoints_orig))
        print('number of simplified lines: {}\n'.format(npoints_simp))
        if npoints_simp > self.maxlines:
            print("Warning, the number of lines exceeds GFLOW's limit of {}!".format(self.maxlines))

        if self.split_by_HUC:
            self.write_lss_by_huc(df)
        else:
            self.write_lss(df, '{}.lss.xml'.format(self.outfile_basename))

        # write shapefile of results
        # convert lists in dn and upcomid columns to strings (for writing to shp)
        self.df = df
        self.write_shapefile()
        self.efp.close()
        print('Done!')

    def map_confluences(self):

        upsegs = self.df.upcomids.tolist()
        maxsegs = np.array([np.max(u) if len(u) > 0 else 0 for u in upsegs])
        seglengths = np.array([len(u) for u in upsegs])
        # setup dataframe of confluences
        # confluences are where segments have upsegs (no upsegs means the reach 1 is a headwater)
        confluences = self.df.ix[(seglengths > 0) & (maxsegs > 0), ['COMID', 'upcomids']].copy()

        confluences['elev'] = [0] * len(confluences)
        nconfluences = len(confluences)
        print('Mapping {} confluences and updating segment min/max elevations...'.format(nconfluences))
        for i, r in confluences.iterrows():

            # confluence elevation is the minimum of the ending segments minimums, starting segments maximums
            endsmin = np.min(self.df.ix[self.df.COMID.isin(r.upcomids), 'minElev'].values)
            startmax = np.max(self.df.ix[self.df.COMID == i, 'maxElev'].values)
            cfelev = np.min([endsmin, startmax])
            confluences.loc[i, 'elev'] = cfelev

            upcomids = [u for u in r.upcomids if u > 0]
            if len(upcomids) > 0:
                self.df.loc[upcomids, 'minElev'] = cfelev
            self.df.loc[i, 'maxElev'] = cfelev

        self.confluences = confluences
        self.df['dStage'] = self.df['maxElev'] - self.df['minElev']
        print('Done, see confluences attribute.')

    def run_diagnostics(self):

        dg = Diagnostics(lsm_object=self)
        dg.check_vertices()
        dg.check4crossing_lines()
        dg.check4zero_gradient()

    def map_outsegs(self):
        '''
        from Mat2, returns dataframe of all downstream segments (will not work with circular routing!)
        '''
        outsegsmap = pd.DataFrame(self.df.COMID)
        outsegs = pd.Series([d[0] if len(d) > 0 else 0 for d in self.df.dncomid], index=self.df.index)
        max_outseg = np.max(outsegsmap[outsegsmap.columns[-1]])
        knt = 2
        while max_outseg > 0:
            outsegsmap['outseg{}'.format(knt)] = [outsegs[s] if s > 0 else 0
                                                  for s in outsegsmap[outsegsmap.columns[-1]]]
            max_outseg = np.max(outsegsmap[outsegsmap.columns[-1]].values)
            if max_outseg == 0:
                break
            knt += 1
            if knt > 1000:
                print('Circular routing encountered in segment {}'.format(max_outseg))
                break
        self.outsegs = outsegsmap

    def read_lss(self, lss_xml):
        """read a linesink string (lss) XML file exported by GFLOW"""
        xml = ET.parse(lss_xml)
        root = xml.getroot()
        for attr in ['ComputationalUnits', 'BasemapUnits']:
            self.__dict__[attr] = root.findall(attr)[0].text

        # read fields into dictionary, then DataFrame
        d = {}
        for field in ['Label',
                      'HeadSpecified',
                      'StartingHead',
                      'EndingHead',
                      'Resistance',
                      'Width',
                      'Depth',
                      'Routing',
                      'EndStream',
                      'OverlandFlow',
                      'EndInflow',
                      'ScenResistance',
                      'Drain',
                      'ScenFluxName',
                      'Gallery',
                      'TotalDischarge',
                      'InletStream',
                      'OutletStream',
                      'OutletTable',
                      'Lake',
                      'Precipitation',
                      'Evapotranspiration',
                      'Farfield',
                      'chkScenario',
                      'AutoSWIZC',
                      'DefaultResistance']:
            if self.dtypes[field] == bool:
                try:
                    v = np.array([i.text for i in root.iter(field)],
                                 dtype=self.int_dtype)
                    d[field] = v.astype(bool)
                except:
                    d[field] = np.array([self.tf2flag(i.text) for i in root.iter(field)])

            else:
                d[field] = np.array([i.text for i in root.iter(field)],
                                dtype=self.dtypes[field])

        # read in the vertices
        def tolss(ls):
            X = np.array([v.find('X').text for v in ls.find('Vertices').findall('Vertex')], dtype=float)
            Y = np.array([v.find('Y').text for v in ls.find('Vertices').findall('Vertex')], dtype=float)
            return LineString(zip(X, Y))
        d['geometry'] = [tolss(ls) for ls in root.iter('LinesinkString')]

        df = pd.DataFrame(d)
        # convert labels back to COMIDs for the index; assign unique values otherwise
        comids = []
        for i, r in df.iterrows():
            try:
                comids.append(int(r.Label.split()[0].strip()))
            except ValueError:
                comids.append(i)
        df.index = comids

        # rename fields to be consistent with those used in Linesinkmaker
        # (should eventually just change lsmaker names to conform)
        df.rename(columns={'DefaultResistance': 'resistance',
                           'Depth': 'depth',
                           'EndStream': 'end_stream',
                           'EndingHead': 'minElev',
                           'Farfield': 'farfield',
                           'Label': 'ls_name',
                           'Resistance': 'reistance',
                           'Routing': 'routing',
                           'StartingHead': 'maxElev',
                           'Width': 'width'}, inplace=True)
        df['ls_coords'] = [list(g.coords) for g in df.geometry]
        return df


    def write_lss_by_huc(self, df):

        print('\nGrouping segments by hydrologic unit...')
        # intersect lines with HUCs; then group dataframe by HUCs
        HUCs_df = GISio.shp2df(self.HUC_shp, index=self.HUC_name_field)
        df[self.HUC_name_field] = len(df) * [None]
        for HUC in HUCs_df.index:
            lines = [line.intersects(HUCs_df.ix[HUC, 'geometry']) for line in df['geometry']]
            df.loc[lines, self.HUC_name_field] = HUC
        dfg = df.groupby(self.HUC_name_field)

        # write lines for each HUC to separate lss file
        HUCs = np.unique(df.HUC)
        for HUC in HUCs:
            dfh = dfg.get_group(HUC)
            outfile = '{}_{}.lss.xml'.format(self.outfile_basename, HUC)
            self.write_lss(dfh, outfile)

    def write_lss(self, df, outfile):
        """write GFLOW linesink XML (lss) file from dataframe df
        """
        df = df.copy()

        nlines = sum([len(p) - 1 for p in df.ls_coords])

        print('writing {} lines to {}'.format(nlines, outfile))
        ofp = open(outfile, 'w')
        ofp.write('<?xml version="1.0"?>\n')
        ofp.write('<LinesinkStringFile version="1">\n')
        ofp.write('\t<ComputationalUnits>{}</ComputationalUnits>\n'
                  '\t<BasemapUnits>{}</BasemapUnits>\n\n'.format(self.ComputationalUnits, self.BasemapUnits))

        for comid in df.index:
            ofp.write('\t<LinesinkString>\n')
            ofp.write('\t\t<Label>{}</Label>\n'.format(df.ix[comid, 'ls_name']))
            ofp.write('\t\t<HeadSpecified>1</HeadSpecified>\n')
            ofp.write('\t\t<StartingHead>{:.2f}</StartingHead>\n'.format(df.ix[comid, 'maxElev']))
            ofp.write('\t\t<EndingHead>{:.2f}</EndingHead>\n'.format(df.ix[comid, 'minElev']))
            ofp.write('\t\t<Resistance>{}</Resistance>\n'.format(df.ix[comid, 'resistance']))
            ofp.write('\t\t<Width>{:.2f}</Width>\n'.format(df.ix[comid, 'width']))
            ofp.write('\t\t<Depth>{:.2f}</Depth>\n'.format(df.ix[comid, 'depth']))
            ofp.write('\t\t<Routing>{}</Routing>\n'.format(df.ix[comid, 'routing']))
            ofp.write('\t\t<EndStream>{}</EndStream>\n'.format(df.ix[comid, 'end_stream']))
            ofp.write('\t\t<OverlandFlow>0</OverlandFlow>\n')
            ofp.write('\t\t<EndInflow>0</EndInflow>\n')
            ofp.write('\t\t<ScenResistance>{}</ScenResistance>\n'.format(df.ix[comid, 'ScenResistance']))
            ofp.write('\t\t<Drain>0</Drain>\n')
            ofp.write('\t\t<ScenFluxName>__NONE__</ScenFluxName>\n')
            ofp.write('\t\t<Gallery>0</Gallery>\n')
            ofp.write('\t\t<TotalDischarge>0</TotalDischarge>\n')
            ofp.write('\t\t<InletStream>0</InletStream>\n')
            ofp.write('\t\t<OutletStream>0</OutletStream>\n')
            ofp.write('\t\t<OutletTable>__NONE__</OutletTable>\n')
            ofp.write('\t\t<Lake>0</Lake>\n')
            ofp.write('\t\t<Precipitation>0</Precipitation>\n')
            ofp.write('\t\t<Evapotranspiration>0</Evapotranspiration>\n')
            ofp.write('\t\t<Farfield>{:.0f}</Farfield>\n'.format(df.ix[comid, 'farfield']))
            ofp.write('\t\t<chkScenario>true</chkScenario>\n')  # include linesink in PEST 'scenarios'
            ofp.write('\t\t<AutoSWIZC>{:.0f}</AutoSWIZC>\n'.format(df.ix[comid, 'AutoSWIZC']))
            ofp.write('\t\t<DefaultResistance>{:.2f}</DefaultResistance>\n'.format(df.ix[comid, 'resistance']))
            ofp.write('\t\t<Vertices>\n')

            # now write out linesink vertices
            for x, y in df.ix[comid, 'ls_coords']:
                ofp.write('\t\t\t<Vertex>\n')
                ofp.write('\t\t\t\t<X> {:.2f}</X>\n'.format(x))
                ofp.write('\t\t\t\t<Y> {:.2f}</Y>\n'.format(y))
                ofp.write('\t\t\t</Vertex>\n')

            ofp.write('\t\t</Vertices>\n')
            ofp.write('\t</LinesinkString>\n\n')
        ofp.write('</LinesinkStringFile>')
        ofp.close()

    def write_shapefile(self, outfile=None, use_ls_coords=True):
        if outfile is None:
            outfile = self.outfile_basename.split('.')[0] + '.shp'
        df = self.df.copy()

        for routid in {'dncomid', 'upcomids'}.intersection(set(df.columns)):
            df[routid] = df[routid].map(lambda x: ' '.join([str(c) for c in x]))  # handles empties

        # recreate shapely geometries from coordinates column; drop all other coords/geometries
        if use_ls_coords and 'ls_coords' in df.columns:
            df['geometry'] = [LineString(g) for g in df.ls_coords]
            df = df.drop(['ls_coords'], axis=1)

        GISio.df2shp(df, outfile, proj4=self.crs_str)

class InputFileMissing(Exception):
    def __init__(self, infile):
        self.infile = infile

    def __str__(self):
        return ('\n\nCould not open or parse input file {0}.\nCheck for errors in XML formatting.'.format(self.infile))