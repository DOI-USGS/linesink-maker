import xml.etree.ElementTree as ET
import sys
import warnings
import numpy as np
import os
from pathlib import Path
import pandas as pd
import shutil
import requests
import json
import yaml
from functools import partial
import fiona
from fiona.crs import to_string, from_string
from shapely.geometry import Polygon, LineString, Point, shape, mapping
from shapely.ops import unary_union, transform
import pyproj
import math
import gisutils
import lsmaker
try:
    import matplotlib.pyplot as plt
except:
    pass
from lsmaker.diagnostics import Diagnostics


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
    """Add vertices to LinesinkData at locations of testpoints
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
            df.loc[ind, 'geometry'] = newline
            #df.set_value(ind, 'geometry', newline)
    return df.geometry.tolist()


def get_elevations_from_epqs(points, units='feet'):
    """From list of shapely points in lat, lon, returns list of elevation values
    """
    if len(points) > 0:
        print('querying National Map Elevation Point Query Service...')
        elevations = [get_elevation_from_epqs(p.x, p.y, units=units) for p in points]
    else:
        elevations = []
    return elevations


def get_elevation_from_epqs(lon, lat, units='feet'):
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
    try:
        #epqsdata = urlopen(url).readline()
        response = requests.get(url)
        elev = json.loads(response.text)['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']
        print(lat, lon, elev, units)
    except:
        e = sys.exc_info()
        print(e)
        print('Problem accessing Elevation Point Query Service. '
              'Need an internet connection to get seepage lake elevations.'
              "\nIf your internet is working, running the script again may work; sometime the EPQS can be temperamental")
        elev = 0.0
    try:
        elev = float(elev)
    except:
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


#def reproject(geoms, pr1, pr2):
#    """Reprojects a list of geometries from coordinate system pr1 to pr2
#    (given as proj strings)."""
#    print('reprojecting from {} to {}...'.format(pr1, pr2))
#    if not isinstance(geoms, list):
#        geoms = [geoms]
#    pr1 = pyproj.Proj(pr1, errcheck=True, preserve_units=True)
#    pr2 = pyproj.Proj(pr2, errcheck=True, preserve_units=True)
#    project = partial(pyproj.transform, pr1, pr2)
#    return [transform(project, g) for g in geoms]


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
    """Abbreviations for naming LinesinkData from names in NHDPlus
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
               'West': 'W',
               "'": ''}

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


class LinesinkData:
    maxlines = 4000

    int_dtype = str  # np.int64

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

    def __init__(self, infile=None, GFLOW_lss_xml=None):

        # attributes
        self._lsmaker_config_file_path = None  # absolute path to config file
        self.preproc = None
        self.resistance = None
        self.H = None  # aquifer thickness in model units
        self.k = None  # hydraulic conductivity of the aquifer in model units
        self.lmbda = None
        self.ScenResistance = None
        self.chkScenario = None
        self.global_streambed_thickness = None  # streambed thickness
        self.ComputationalUnits = None  # 'feet' or 'meters'; for XML output file
        self.BasemapUnits = None
        # elevation units multiplier (from NHDPlus cm to model units)
        self.zmult = None

        # model domain
        self.farfield = None
        self.routed_area = None
        self.nearfield = None
        self.prj = None
        self.crs = None
        #self.crs_str = None  # self.crs, in proj string format
        self.pyproj_crs = None  # pyproj.CRS instance based on prj input
        self.farfield_buffer = None
        self.clip_farfield = None
        self.split_by_HUC = None
        self.HUC_shp = None
        self.HUC_name_field = None

        # simplification
        self.refinement_areas = []  # list of n areas within routed area with additional refinement
        self.nearfield_tolerance = None
        self.routed_area_tolerance = None
        self.farfield_tolerance = None
        self.min_nearfield_order = None
        self.min_routed_area_order = None
        self.min_farfield_order = None
        self.min_nearfield_wb_size = None
        self.min_waterbody_size = None
        self.min_farfield_wb_size = None
        self.farfield_length_threshold = None
        self.routed_area_length_threshold = None
        self.drop_intermittent = None
        self.drop_crossing = None
        self.asum_thresh_ra = None
        self.asum_thresh_nf = None
        self.asum_thresh_ff = None

        # NHD files
        self.flowlines = None
        self.elevslope = None
        self.PlusFlowVAA = None
        self.waterbodies = None

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
        self.DEM = None
        self.elevs_field = None
        self.DEM_zmult = None

        self.flowlines_clipped = None
        self.waterbodies_clipped = None
        self.routed_mp = None
        self.farfield_mp = None
        self.preprocessed_lines = None
        self.preprocdir = None
        self.wb_centroids_w_elevations = None  # elevations extracted during preprocessing routine
        self.elevs_field = None  # field in wb_centroids_w_elevations containing elevations

        # outputs
        self.outfile_basename = None
        self.error_reporting = 'error_reporting.txt'

        # attributes
        self.from_lss_xml = False
        self.df = pd.DataFrame() # working dataframe for translating NHDPlus data to linesink strings
        self.lss = pd.DataFrame() # dataframe of GFLOW linesink strings (with attributes)
        self.outsegs = pd.DataFrame()
        self.confluences = pd.DataFrame()

        # read in the configuration file
        if infile is not None and infile.endswith('.xml'):
            self.read_lsmaker_xml(infile)
        if infile is not None:
            for extension in 'yml', 'yaml':
                if infile.endswith(extension):
                    self.read_lsmaker_yaml(infile)

        # or create instance from a GFLOW LSS XML file
        elif GFLOW_lss_xml is not None:
            self.from_lss_xml = True
            self.df = self.read_lss(GFLOW_lss_xml)
            
        # nearfield can't be coarser than the routed area
        if infile is not None:
            self.min_nearfield_order = min((self.min_nearfield_order, 
                                            self.min_routed_area_order))
        # create a pyproj CRS instance
        # set the CRS (basemap) length units
        self.set_crs(prjfile=self.prj)

        # logging/diagnostics
        # todo: more robust/detail logging
        if Path(self.error_reporting).exists():
            try:
                Path(self.error_reporting).unlink()
            except:
                j=2
        with open(self.error_reporting, 'w') as efp:
            efp.write('Linesink-maker version {}\n'.format(lsmaker.__version__))
        self.dg = Diagnostics(lsm_object=self, logfile=self.error_reporting)

    def __eq__(self, other):
        """Test for equality to another linesink object."""
        if not isinstance(other, self.__class__):
            return False
        exclude_attrs = ['_lsmaker_config_file_path',
                         'crs',
                         'crs_str',
                         'inpars',
                         'cfg',
                         'efp',
                         ]
        # LinesinkData instances from lss xml won't have some attributes
        # or some df columns that came from NHDPlus
        compare_df_columns = slice(None)
        if self.from_lss_xml or other.from_lss_xml:
            # todo: expose default values that are being set on write of lss_xml
            # (many of the columns in LinesinkData instance from lss xml aren't in
            # LinesinkData instance that was created from scratch, because the variables
            # are only being set in LinesinkData.write_lss method)
            compare_df_columns = set(self.df.columns).intersection(other.df.columns)
            # the geometries and coordinates won't be exactly the same
            # explicitly compare the coordinates separately
            compare_df_columns = compare_df_columns.difference({'geometry',
                                                                'ls_coords',
                                                                'width'
                                                                })
        for k, v in self.__dict__.items():
            # items to skip
            # todo: implement pyproj.CRS class to robustly compare CRSs
            if k in exclude_attrs:
                continue
            elif self.from_lss_xml or other.from_lss_xml:
                if k not in ('df', 'ComputationalUnits', 'BasemapUnits'):
                    continue
            elif k not in other.__dict__:
                return False
            elif type(v) == bool:
                if not v == other.__dict__[k]:
                    return False
            elif k == 'df':
                if len(v) == 0 and len(other.__dict__[k]) == 0:
                    continue
                try:
                    df1 = v[compare_df_columns]
                    df2 = other.__dict__[k][compare_df_columns]
                    #
                    pd.testing.assert_frame_equal(df1, df2)
                    # compare the coordinates
                    for dim in 0, 1:  # (x, y)
                        x1 = [crd[dim] for line in v.ls_coords for crd in line]
                        x2 = [crd[dim] for line in other.__dict__[k].ls_coords for crd in line]
                        assert np.allclose(x1, x2)
                    assert np.allclose(v.width.values, other.__dict__[k].width.values, rtol=0.01)
                except:
                    return False
            elif type(v) == pd.DataFrame:
                try:
                    pd.testing.assert_frame_equal(v, other.__dict__[k])
                except:
                    return False

            elif v != other.__dict__[k]:
                try:
                    if not np.allclose(v, other.__dict__[k]):
                        return False
                except:
                    continue
                    #return False
            #elif type(v) in [str, int, float, dict, list]:
            #    if v != other.__dict__[k]:
            #        pass
            #    continue
        
    def read_lsmaker_xml(self, infile):

        #try:
        inpardat = ET.parse(infile)
        #except:
        #    raise InputFileMissing

        # record the config file absolute path
        self._lsmaker_config_file_path = os.path.split(os.path.abspath(infile))[0]

        inpars = inpardat.getroot()
        self.inpars = inpars

        # setup the working directory (default to current directory)
        try:
            self.path = inpars.findall('.//working_dir')[0].text
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        except:
            self.path = self._lsmaker_config_file_path

        # global settings
        self.preproc = self.tf2flag(self._get_xml_entry('preproc', 'True'))
        self.resistance = self._get_xml_entry('resistance', 0.3, float)  # (days); c in documentation
        self.H = self._get_xml_entry('H', 100, float) # aquifer thickness in model units
        self.k = self._get_xml_entry('k', 10, float)  # hydraulic conductivity of the aquifer in model units
        self.lmbda = np.sqrt(self.k * self.H * self.resistance)
        self.ScenResistance = self._get_xml_entry('ScenResistance', 'linesink')
        self.chkScenario = self.tf2flag(self._get_xml_entry('chkScenario', 'True'))
        self.global_streambed_thickness = self._get_xml_entry('global_streambed_thickness',
                                                              3, float)  # streambed thickness
        self.ComputationalUnits = self._get_xml_entry('ComputationalUnits', 'feet').lower() # 'feet' or 'meters'; for XML output file
        self.BasemapUnits = self._get_xml_entry('BasemapUnits', None)
        if self.BasemapUnits is not None:
            self.BasemapUnits = self.BasemapUnits.lower()
        # elevation units multiplier (from NHDPlus cm to model units)
        self.zmult = 0.03280839895013123 if self.ComputationalUnits.lower() == 'feet' else 0.01

        # model domain
        self.farfield = self._get_xml_entry('farfield', None)
        self.routed_area = self._get_xml_entry('routed_area', None)
        self.nearfield = self._get_xml_entry('nearfield', None)
        try: # get the projection file and crs from either the nearfield or routed area
            self.prj = self._get_xml_entry('prj', self.nearfield[:-4] + '.prj')
            self.crs = fiona.open(self.nearfield).crs
        except:
            self.prj = self._get_xml_entry('prj', self.routed_area[:-4] + '.prj')
            self.crs = fiona.open(self.routed_area).crs
            self.nearfield = self.routed_area
        #self.crs_str = str(self.crs)  # self.crs, in proj string format
        self.farfield_buffer = self._get_xml_entry('farfield_buffer', 10000, int)
        self.clip_farfield = self.tf2flag(self._get_xml_entry('clip_farfield', 'False'))
        self.split_by_HUC = self.tf2flag(self._get_xml_entry('split_by_HUC', 'False'))
        self.HUC_shp = self._get_xml_entry('HUC_shp', None)
        self.HUC_name_field = self.tf2flag(self._get_xml_entry('HUC_name_field', 'False'))

        # simplification
        self.refinement_areas = []  # list of n areas within routed area with additional refinement
        self.nearfield_tolerance = self._get_xml_entry('nearfield_tolerance', 100, int)
        self.routed_area_tolerance = self._get_xml_entry('routed_area_tolerance', 100, int)
        self.farfield_tolerance = self._get_xml_entry('farfield_tolerance', 300, int)
        self.min_farfield_order = self._get_xml_entry('min_farfield_order', 2, int)
        self.min_nearfield_order = self._get_xml_entry('min_nearfield_order', 1, int)
        self.min_routed_area_order = self._get_xml_entry('min_routed_area_order', 1, int)
        self.min_nearfield_wb_size = self._get_xml_entry('min_nearfield_waterbody_size', 1.0, float)
        self.min_waterbody_size = self._get_xml_entry('min_waterbody_size', 1.0, float)
        self.min_farfield_wb_size = self._get_xml_entry('min_farfield_waterbody_size',
                                                        self.min_waterbody_size, float)
        self.farfield_length_threshold = self._get_xml_entry('farfield_length_threshold', 0., float)
        self.routed_area_length_threshold = self._get_xml_entry('routed_area_length_threshold', 0., float)
        self.drop_intermittent = self.tf2flag(self._get_xml_entry('drop_intermittent', 'False'))
        self.drop_crossing = self.tf2flag(self._get_xml_entry('drop_crossing', 'False'))
        self.asum_thresh_ra = self._get_xml_entry('routed_area_arbolate_sum_threshold', 0., float)
        self.asum_thresh_nf = self._get_xml_entry('nearfield_arbolate_sum_threshold', 0., float)
        self.asum_thresh_ff = self._get_xml_entry('farfield_arbolate_sum_threshold', 0., float)

        # NHD files
        self.flowlines = self._get_xml_entry('flowlines', [], str, raise_error=True)
        self.elevslope = self._get_xml_entry('elevslope', [], str, raise_error=True)
        self.PlusFlowVAA = self._get_xml_entry('PlusFlowVAA', [], str, raise_error=True)
        self.waterbodies = self._get_xml_entry('waterbodies', [], str, raise_error=True)

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
        self.DEM = self._get_xml_entry('DEM', None)
        self.elevs_field = self._get_xml_entry('elevs_field', 'elev')
        self.DEM_zmult = self._get_xml_entry('DEM_zmult', 1.0, float)

        self.flowlines_clipped = self._get_xml_entry('flowlines_clipped',
                                                    'preprocessed/flowlines_clipped.shp',
                                                     relative_filepath=True)
        self.waterbodies_clipped = self._get_xml_entry('waterbodies_clipped',
                                                      'preprocessed/waterbodies_clipped.shp',
                                                       relative_filepath=True)
        self.routed_mp = self._get_xml_entry('routed_area_multipolygon',
                                            'preprocessed/ra_cutout.shp',
                                             relative_filepath=True)
        self.farfield_mp = self._get_xml_entry('farfield_multipolygon',
                                              'preprocessed/ff_cutout.shp',
                                               relative_filepath=True)
        self.preprocessed_lines = self._get_xml_entry('preprocessed_lines',
                                                     'preprocessed/lines.shp',
                                                      relative_filepath=True)
        self.preprocdir = os.path.split(self.flowlines_clipped)[0]

        self.wb_centroids_w_elevations = self.waterbodies_clipped[
                                         :-4] + '_points.shp'  # elevations extracted during preprocessing routine
        self.elevs_field = 'DEM'  # field in wb_centroids_w_elevations containing elevations

        # outputs
        self.outfile_basename = os.path.join(self._lsmaker_config_file_path,
                                             inpars.findall('.//outfile_basename')[0].text)
        self.error_reporting = os.path.join(self._lsmaker_config_file_path,
                                            inpars.findall('.//error_reporting')[0].text)

        # attributes
        self.df = pd.DataFrame() # working dataframe for translating NHDPlus data to linesink strings
        self.lss = pd.DataFrame() # dataframe of GFLOW linesink strings (with attributes)
        self.outsegs = pd.DataFrame()
        self.confluences = pd.DataFrame()

    def read_lsmaker_yaml(self, infile):

        # read in the user-specified configuration
        with open(infile) as src:
            cfg = yaml.load(src, Loader=yaml.Loader)
        # read in the default configuration
        defaults_file = os.path.join(os.path.split(__file__)[0], 
                                     'default_settings.yml')
        with open(defaults_file) as src:
            defaults = yaml.load(src, Loader=yaml.Loader)

        self._lsmaker_config_file_path = os.path.split(os.path.abspath(infile))[0]

        # configuration dictionary
        self.cfg = cfg

        # configuration file blocks
        filepaths_to_make_abs = {'outfile_basename'}
        for blockname, block in defaults.items():
            # entries within blocks
            for key, default in defaults[blockname].items():
                # try to detect files and make their paths absolute
                entry = self.cfg.get(blockname, {}).get(key, default)
                if isinstance(entry, str):
                    file_abspath = os.path.join(self._lsmaker_config_file_path, entry)
                    if os.path.exists(file_abspath) or key in filepaths_to_make_abs:
                        entry = file_abspath
                self.__dict__[key] = entry
                
        # setup the working directory (default to current directory)
        self.path = os.path.abspath(self.__dict__.pop('working_dir'))
        
        self.lmbda = np.sqrt(self.k * self.H * self.resistance)
        self.zmult = 0.03280839895013123 if self.ComputationalUnits.lower() == 'feet' else 0.01
            
        # get the projection file and crs from either the nearfield or routed area
        if self.prj is None:
            for filename in self.nearfield, self.routed_area:
                if filename is not None:
                    name, ext = os.path.splitext(filename)
                    prjfile = name + '.prj'
                    if os.path.exists(prjfile):
                        self.prj = prjfile
                        self.crs = gisutils.get_shapefile_crs(prjfile)
                        #with fiona.open(filename) as src:
                        #    self.crs = src.crs
                        #    self.crs_str = to_string(self.crs)
        else:
            #self.crs_str = gisutils.get_proj_str(self.prj)
            self.crs = gisutils.get_shapefile_crs(self.prj)
        if self.crs is None or self.prj is None:
            msg = ("Invalid projection file or projection file not found: {}. \
                Specify a valid ESRI projection file under the \
                prj: configuration file key".format(self.prj))
            raise ValueError(msg)

        # NHDPlus files
        for variable in 'flowlines', 'elevslope', 'PlusFlowVAA', 'waterbodies':
            filename = self.__dict__[variable]
            if filename is None:
                raise KeyError('Nothing specified for {} in the configuration file!'.format(variable))
            elif not os.path.exists(filename):
                raise ValueError('file not found: {}'.format(filename))

        # preprocessed files
        self.preprocdir = os.path.split(self.flowlines_clipped)[0]
        self.wb_centroids_w_elevations = self.waterbodies_clipped[
                                         :-4] + '_points.shp'  # elevations extracted during preprocessing routine
        self.elevs_field = 'DEM'  # field in wb_centroids_w_elevations containing elevations
        
    def _parse_xml_text(self, findall_result, dtype, relative_filepath=False):
        # handle either strings or lxml Elements
        txt = getattr(findall_result, 'text', findall_result)
        if dtype == str:
            # check if a file exists relative to config file path
            # if so, change the file path to be absolute
            file_abspath = os.path.join(self._lsmaker_config_file_path, txt)
            if os.path.exists(file_abspath) or relative_filepath:
                txt = file_abspath
        return dtype(txt) if txt is not None else None

    def _get_xml_entry(self, XMLentry, default, dtype=str,
                       relative_filepath=False,
                       raise_error=False):
        try:
            tmp = self.inpars.findall('.//{}'.format(XMLentry))
            if len(tmp) == 0:
                if not raise_error and not relative_filepath:
                    return default
                elif not raise_error and relative_filepath:
                    tmp = [default] if not isinstance(default, list) else default
                else:
                    raise ValueError('Nothing specified for {} in input XML file!'.format(XMLentry))
            if len(tmp) == 1: # and not isinstance(default, list):
                return self._parse_xml_text(tmp[0], dtype, relative_filepath=relative_filepath)
            elif len(tmp) >= 1:
                return [self._parse_xml_text(s, dtype, relative_filepath=relative_filepath)
                        for s in tmp]
                #return [dtype(s.text) for s in tmp]
            else:
                raise Exception()
        except:
            if not raise_error:
                return default
            else:
                raise ValueError('Nothing specified for {} in input XML file!'.format(XMLentry))
            
    def _enforce_dtypes(self, df):
        """Ensure that dataframe column dtypes are correct."""
        for c in df.columns:
            dtype = self.dtypes.get(c, df[c].dtype)
            try:
                df[c] = df[c].astype(dtype)
            except ValueError:
                # handle Nonetype values in some NHDPlus fields
                j=2
                continue

    def _get_wb_elevations(self, wb_df):
        # (the centroids of some lakes might not be in the lake itself)
        wb_points = [wb.centroid if wb.centroid.within(wb)
                     else _get_random_point_in_polygon(wb)
                     for wb in wb_df.geometry.tolist()]
        # reproject the representative points into lat lon
        wb_points_ll = gisutils.project(wb_points, self.crs, "epsg:4269")
        wb_elevations = get_elevations_from_epqs(wb_points_ll, units=self.ComputationalUnits)

        # add in elevations from waterbodies in stream network

        return pd.DataFrame({'COMID': wb_df.COMID.values,
                                     self.elevs_field: wb_elevations,
                                     'geometry': wb_points})

    def load_poly(self, shapefile, dest_crs=None):
        """Load a shapefile and return the first polygon from its records,
        projected in the destination coordinate system."""
        if dest_crs is None:
            dest_crs = self.crs
        else:
            dest_crs = gisutils.get_authority_crs(dest_crs)
        print('reading {}...'.format(shapefile))
        with fiona.open(shapefile) as src:
            geom = shape(next(iter(src))['geometry'])
        shp_crs = gisutils.get_shapefile_crs(shapefile)
        if shp_crs != dest_crs:
            geom = gisutils.project(geom, shp_crs, dest_crs)[0]
        return geom

    def tf2flag(self, intxt):
        # converts text written in XML file to True or False flag
        if intxt.lower() == 'true':
            return True
        elif intxt.lower() == 'none':
            return
        else:
            return False

    def set_crs(self, epsg=None, proj_str=None, prjfile=None):
        """Set the projected coordinate reference system, and the BasemapUnits.
        If no arguments are supplied, default to existing BasemapUnits.
        
        Parameters
        ----------
        epsg: int
            EPSG code identifying Coordinate Reference System (CRS)
            for features in df.geometry
            (optional)
        proj_str: str
            proj_str string identifying CRS for features in df.geometry
            (optional)
        prjfile: str
            File path to projection (.prj) file identifying CRS
            for features in df.geometry
            (optional)
        
        Returns
        -------
        sets the LinesinkData.pyproj_crs attribute.
        """
        args = any(arg for arg in (epsg, proj_str, prjfile))
        pyproj_crs = None
        if self.BasemapUnits is None or args:
            if epsg is not None:
                pyproj_crs = pyproj.CRS.from_epsg(epsg)
            elif proj_str is not None:
                pyproj_crs = pyproj.CRS.from_string(proj_str)
            elif prjfile is not None:
                with open(prjfile) as src:
                    wkt = src.read()
                pyproj_crs = pyproj.CRS.from_wkt(wkt)
            # if possible, have pyproj try to find the closest
            # authority name and code matching the crs
            # so that input from epsg codes, proj strings, and prjfiles
            # results in equal pyproj_crs instances
            if pyproj_crs is not None:
                try:
                    authority = pyproj_crs.to_authority()
                    if authority is not None:
                        self.pyproj_crs = pyproj.CRS.from_user_input(pyproj_crs.to_authority())
                    else:
                        self.pyproj_crs = pyproj_crs
                except:
                    pass
                units = pyproj_crs.axis_info[0].unit_name
                translate_units = {'metre': 'meters'}
                self.BasemapUnits = translate_units.get(units, units)
            
    def preprocess(self, save=True):
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
            self.routed_area_tolerance = self.nearfield_tolerance
        if self.nearfield is None and self.routed_area is None:
            raise InputFileMissing('Need to supply shapefile of routed area or nearfield.')

        if self.farfield is None:
            print(('\nNo farfield shapefile supplied.\n'
                  'Creating farfield using buffer of {:.1f} {} around routed area.\n'
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

        # make the output directory if it doesn't exist yet
        if len(self.preprocdir) > 0 and not os.path.isdir(self.preprocdir):
            os.makedirs(self.preprocdir)

        # make projection file that is independent of any shapefile
        if os.path.exists('GFLOW.prj'):
            os.remove('GFLOW.prj')
        if self.prj is not None and os.path.exists(self.prj):
            shutil.copy(self.prj, 'GFLOW.prj')
        self.prj = 'GFLOW.prj'

        print('clipping and reprojecting input datasets...')

        # (zero buffers can clean up self-intersections in hand drawn polygons)
        self.nf = self.load_poly(self.nearfield).buffer(0)
        self.ra = self.load_poly(self.routed_area).buffer(0)
        self.ff = self.load_poly(self.farfield).buffer(0)
        for p in [self.nf, self.ra, self.ff]:
            assert p.is_valid, 'Invalid polygon'

        for attr, shapefiles in list({'fl': self.flowlines,
                                      'wb': self.waterbodies}.items()):
            # all flowlines and waterbodies must be in same coordinate system
            # sfrmaker preproc also uses crs from first file in list
            if isinstance(shapefiles, list):
                shapefile = shapefiles[0]
            else:
                shapefile = shapefiles
            shp_crs = gisutils.get_shapefile_crs(shapefile)

            # get bounding box of model area in nhd crs to speeding reading in data via filter
            bounds = gisutils.project([self.ff], self.crs, shp_crs)
            bounds = bounds[0].bounds

            self.__dict__[attr] = gisutils.shp2df(shapefiles, filter=bounds)
            # if NHD features not in model area coodinate system, reproject
            if shp_crs != self.crs:
                self.__dict__[attr]['geometry'] = gisutils.project(self.__dict__[attr].geometry.tolist(), shp_crs,
                                                            self.crs)
            # 1816107
            # for now, take intersection (don't truncate flowlines at farfield boundary)
            print('clipping to {}...'.format(self.farfield))
            intersects = np.array([l.intersects(self.ff) for l in self.__dict__[attr].geometry])
            self.__dict__[attr] = self.__dict__[attr][intersects].copy()
        if self.clip_farfield:
            print('truncating waterbodies at farfield boundary...')
            # get rid of any islands in the process
            wbgeoms = self.wb.geometry
            # valid = np.array([p.is_valid for p in wbgeoms])
            # if np.any(~valid):
            #    comids = self.wb.loc[~valid, 'COMID']
            #    print('The following waterbodies result in invalid polygons when intersected with the domain:')
            #    for c in comids:
            #        print(c)
            wbgeoms = [Polygon(g.exterior).buffer(0)
                       for g in self.wb.geometry]
            wbgeoms = [p.intersection(self.ff).buffer(0) for p in wbgeoms]
            self.wb['geometry'] = wbgeoms
            # for multipolygons, retain largest part
            geoms = self.wb.geometry.values
            for i, g in enumerate(geoms):
                if g.type == 'MultiPolygon':
                    geoms[i] = np.array(g.geoms)[np.argsort([p.area for p in g.geoms])][-1]
            self.wb['geometry'] = geoms

        # for now, write out preprocessed data to maintain structure with arcpy
        for df in ['fl', 'wb']:
            if len(self.__dict__[df]) == 0:
                raise EmptyDataFrame()

        gisutils.df2shp(self.fl, self.flowlines_clipped, crs=self.crs)
        gisutils.df2shp(self.wb, self.waterbodies_clipped, crs=self.crs)

        print('\nmaking donut polygon of farfield (with routed area removed)...')
        ffdonut = self.ff.difference(self.ra)
        with fiona.open(self.routed_area) as src:
            with fiona.open(self.farfield_mp, 'w', **src.meta) as output:
                print(('writing {}'.format(self.farfield_mp)))
                f = next(iter(src))
                f['geometry'] = mapping(ffdonut)
                output.write(f)

        if self.routed_area is not None and self.routed_area != self.nearfield:
            print('\nmaking donut polygon of routed area (with nearfield area removed)...')
            if not self.nf.within(self.ra):
                raise ValueError('Nearfield area must be within routed area!')

            donut = self.ra.difference(self.nf)
            with fiona.open(self.nearfield) as src:
                with fiona.open(self.routed_mp, 'w', **src.meta) as output:
                    print(('writing {}'.format(self.routed_mp)))
                    f = next(iter(src))
                    f['geometry'] = mapping(donut)
                    output.write(f)

        # drop waterbodies that aren't lakes bigger than min size
        min_size = np.min([self.min_nearfield_wb_size, self.min_waterbody_size, self.min_farfield_wb_size])
        self.wb = self.wb.loc[(self.wb.FTYPE == 'LakePond') & (self.wb.AREASQKM > min_size)].copy()

        print('\ngetting elevations for waterbodies not in the stream network')
        isolated_wb = self.wb
        isolated_wb_comids = isolated_wb.COMID.tolist()
        if not os.path.exists(self.wb_centroids_w_elevations):
            # get elevations for all waterbodies for the time being.
            # otherwise waterbodies associated with first-order streams may be dropped from farfield,
            # but still waterbodies list, causing a key error in the lakes setup
            # these lakes probably should be left in the farfield unless they are explicitly not wanted
            '''
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
            '''
            wb_points_df = self._get_wb_elevations(isolated_wb)
            gisutils.df2shp(wb_points_df, self.wb_centroids_w_elevations, crs=self.crs)
        else:
            print('\nreading elevations from {}...'.format(self.wb_centroids_w_elevations))
            wb_points_df = gisutils.shp2df(self.wb_centroids_w_elevations)
            toget = set(isolated_wb_comids).difference(set(wb_points_df.COMID.tolist()))
            if len(toget) > 0:
                isolated_wb = self.wb.loc[self.wb.COMID.isin(toget)].copy()
                wb_points_df2 = self._get_wb_elevations(isolated_wb)
                wb_points_df = wb_points_df.append(wb_points_df2)
                gisutils.df2shp(wb_points_df, self.wb_centroids_w_elevations, crs=self.crs)

        # open error reporting file
        with open(self.error_reporting, 'a') as efp:
            efp.write('\nPreprocessing...\n')

        print('\nAssembling input...')
        # read linework shapefile into pandas dataframe
        df = gisutils.shp2df(self.flowlines_clipped, index='COMID', index_dtype=self.int_dtype).drop_duplicates('COMID')
        df.drop([c for c in df.columns if c.lower() not in [cc.lower() for cc in self.flowlines_cols]],
                axis=1, inplace=True)
        # might want to consider enforcing integer index here if strings cause problems
        #clipto = df.index.tolist()
        elevs = gisutils.shp2df(self.elevslope, index='COMID', index_dtype=self.int_dtype)  #, clipto=clipto)
        elevs = elevs.loc[elevs.COMID.isin(df.index)]
        pfvaa = gisutils.shp2df(self.PlusFlowVAA, index='COMID', index_dtype=self.int_dtype)  #, clipto=clipto)
        pfvaa = pfvaa.loc[pfvaa.ComID.isin(df.index)]
        wbs = gisutils.shp2df(self.waterbodies_clipped, index='COMID', index_dtype=self.int_dtype).drop_duplicates('COMID')
        wbs.drop([c for c in wbs.columns if c.lower() not in [cc.lower() for cc in self.wb_cols]],
                 axis=1, inplace=True)
        self._enforce_dtypes(wbs)

        # check for MultiLineStrings / MultiPolygons and drop them (these are features that were fragmented by the boundaries)
        mls = [i for i in df.index if 'Multi' in df.loc[i, 'geometry'].type]
        df = df.drop(mls, axis=0)
        mps = [i for i in wbs.index if 'Multi' in wbs.loc[i, 'geometry'].type]
        wbs = wbs.drop(mps, axis=0)

        # join NHD tables to lines
        df = df.join(elevs[self.elevslope_cols], how='inner', lsuffix='1')
        df = df.join(pfvaa[self.pfvaa_cols], how='inner', lsuffix='1')
        self._enforce_dtypes(df)

        # read in nearfield and farfield boundaries
        nf = gisutils.shp2df(self.nearfield)
        nfg = nf.iloc[0]['geometry']  # polygon representing nearfield
        if self.routed_area is not None and self.routed_area != self.nearfield:
            ra = gisutils.shp2df(self.routed_mp)
            rag = ra.iloc[0]['geometry']
        else:
            rag = nfg
            nfg = Polygon() # no nearfield specified
        ff = gisutils.shp2df(self.farfield_mp)
        ffg = ff.iloc[0]['geometry']  # shapely geometry object for farfield (polygon with interior ring for nearfield)

        print('\nidentifying farfield and nearfield LinesinkData...')
        df['farfield'] = [line.intersects(ffg) and not line.intersects(rag) for line in df.geometry]
        wbs['farfield'] = [poly.intersects(ffg) and not poly.intersects(rag) for poly in wbs.geometry]
        df['routed'] = [line.intersects(rag) for line in df.geometry]
        wbs['routed'] = [poly.intersects(rag) for poly in wbs.geometry]
        df['nearfield'] = [line.intersects(nfg) for line in df.geometry]
        wbs['nearfield'] = [poly.intersects(nfg) for poly in wbs.geometry]

        if self.asum_thresh_ra > 0.:
            print('\nremoving streams in routed area with arbolate sums < {:.2f}'.format(self.asum_thresh_ra))
            # retain all streams in the farfield or in the routed area with arbolate sum > threshold
            criteria = ~df.routed.values | df.routed.values & (df.ArbolateSu.values > self.asum_thresh_ra)
            df = df.loc[criteria].copy()

        if self.asum_thresh_nf > 0.:
            print('\nremoving streams in nearfield with arbolate sums < {:.2f}'.format(self.asum_thresh_nf))
            criteria = ~df.nearfield.values | df.nearfield.values & (df.ArbolateSu.values > self.asum_thresh_nf)
            df = df.loc[criteria].copy()

        if self.asum_thresh_ff > 0.:
            print('\nremoving streams in farfield with arbolate sums < {:.2f}'.format(self.asum_thresh_ff))
            criteria = ~df.farfield.values | df.farfield.values & (df.ArbolateSu.values > self.asum_thresh_ff)
            df = df.loc[criteria].copy()

        if self.min_nearfield_order > 1.:
            print('removing nearfield streams lower than {} order...'.format(self.min_nearfield_order))
            # retain all streams in the nearfield of order > min_farfield_order
            criteria = ~df.nearfield.values | (df.nearfield.values & (df.StreamOrde.values >= self.min_nearfield_order))
            df = df[criteria].copy()
        
        if self.min_routed_area_order > 1.:
            print('removing streams in the routed area of lower than {} order...'.format(self.min_routed_area_order))
            # retain all streams in the nearfield of order > min_farfield_order
            routed = df.routed.values & ~df.nearfield.values
            criteria = ~routed | (routed & (df.StreamOrde.values >= self.min_routed_area_order))
            df = df[criteria].copy()
            
        if self.min_farfield_order > 1.:
            print('removing farfield streams lower than {} order...'.format(self.min_farfield_order))
            # retain all streams in the nearfield or in the farfield and of order > min_farfield_order
            criteria = ~df.farfield.values | (df.farfield.values & (df.StreamOrde.values >= self.min_farfield_order))
            df = df[criteria].copy()

        farfield_retain = ~(df.farfield.values & (df.FCODE == self.fcodes['Intermittent']).values) # drop intermittent streams from farfield
        if self.drop_intermittent:
            print('removing intermittent streams from routed area outside of nearfield...')
            # retain all streams in the nearfield or in the farfield and of order > min_farfield_order
            farfield_retain = farfield_retain & ~(df.routed.values & ~df.nearfield.values & (df.FCODE == self.fcodes['Intermittent']).values)
        df = df[farfield_retain].copy()

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
            overlapping = wbs.loc[[wb_geometry.intersects(r) for r in wbs.geometry]]
            basering_comid = overlapping.sort_values('FTYPE').index[0]  # sort to prioritize features with names

            # two or more shapes in overlapping signifies a coincident boundary
            if len(overlapping) > 1:
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
        wbs['waterbody'] = np.array([True] * len(wbs), dtype=bool)  # boolean variable indicate whether feature is waterbody

        print('merging flowline and waterbody datasets...')
        df['waterbody'] = np.array([False] * len(df), dtype=bool)
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
            gisutils.df2shp(df, self.preprocessed_lines, crs=self.crs)

        self.df = df

    def simplify_lines(self, nearfield_tolerance=None, routed_area_tolerance=None, farfield_tolerance=None,
                       nearfield_refinement={}):
        """Reduces the number of vertices in the GIS linework representing streams and lakes,
        to within specified tolerances. The tolerance values represent the maximum distance
        in the coordinate system units that the simplified feature can deviate from the original feature.

        Parameters
        ----------
        nearfield_tolerance : float
            Tolerance for the area representing the model nearfield
        farfield_tolerance : float
            Tolerance for the area representing the model farfield

        Returns
        -------
        df : DataFrame
            A copy of the df attribute with a 'ls_geom' column of simplified geometries, and
            a 'ls_coords' column containing lists of coordinate tuples defining each simplified line.
        """
        if not hasattr(self, 'df'):
            print('No dataframe attribute for LinesinkData instance. Run preprocess first.')
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

        ls_geom = np.array([LineString()] * len(df), dtype=object)
        tols = {'nf': nearfield_tolerance,
                'ra': routed_area_tolerance,
                'ff': farfield_tolerance}
        simplification = {'nf': df.nearfield.values,
                          'ra': df.routed.values & ~df.nearfield.values & ~df.farfield.values,
                          'ff': df.farfield.values}
        [simplification.pop(k) for k, v in tols.items() if v is None]

        for k, within in simplification.items():
            # simplify the LinesinkData in the domain; add simplified geometries to global geometry column
            # assign geometries to numpy array first and then to df (had trouble assigning with pandas)
            ls_geom[within] = [g.simplify(tols[k]) for g in df.loc[within, 'geometry'].tolist()]

        df['ls_geom'] = ls_geom

        # add columns for additional nearfield refinement areas
        for shp in list(nearfield_refinement.keys()):
            if shp not in self.refinement_areas:
                self.refinement_areas.append(shp)
            area_name = os.path.split(shp)[-1][:-4]
            with fiona.open(shp) as src:
                poly = shape(next(iter(src))['geometry'])
            #poly = shape(fiona.open(shp).next()['geometry'])
            df[area_name] = [g.intersects(poly) for g in df.geometry]

        # add column of lists, containing linesink coordinates
        df['ls_coords'] = [list(g.coords) for g in df.ls_geom]

        assert np.all(np.array([len(c) for c in df.ls_coords]) > 0) # shouldn't be an empty coordinates

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
            gisutils.df2shp(df, outshp, geo_column='ls_geom', crs=self.crs)

        plt.figure()
        plt.plot(nftol, nlines)
        plt.xlabel('Distance tolerance')
        plt.ylabel('Number of lines')
        plt.savefig(self.outfile_basename + 'tol_vs_nlines.pdf')

    def adjust_zero_gradient(self, df, increment=0.01):

        dg = self.dg
        dg.df = df
        comids0 = dg.check4zero_gradient(log=False)

        if len(comids0) > 0:

            if len(self.confluences) == 0:
                self.df = df
                self.map_confluences()
            if len(self.outsegs) == 0:
                self.df = df
                self.map_outsegs()

            with open(self.error_reporting, 'a') as efp:
                efp.write('\nzero-gradient adjustments:\n')
                efp.write('comid, old_elevmax, old_elevmin, new_elevmax, new_elevmin, downcomid\n')

                print("adjusting elevations for comids with zero-gradient...")
                for comid in comids0:

                    outsegs = [o for o in self.outsegs.loc[comid].values if int(o) > 0]
                    for i, o in enumerate(outsegs):

                        if i == len(outsegs) - 1:
                            oo = 0
                        else:
                            oo = outsegs[i + 1]

                        minElev, maxElev = df.loc[o, 'minElev'], df.loc[o, 'maxElev']
                        # test if current segment has flat or negative gradient
                        if minElev >= maxElev:
                            minElev = maxElev - increment
                            df.loc[o, 'minElev'] = minElev
                            efp.write('{}, {:.2f}, {:.2f}, {:.2f}, {}\n'.format(o, maxElev,
                                                                                    minElev + increment,
                                                                                    maxElev,
                                                                                    minElev, oo))
                            # test if next segment is now higher
                            if int(oo) > 0 and df.loc[oo, 'maxElev'] > minElev:
                                efp.write('{}, {:.2f}, {:.2f}, {:.2f}, {}\n'.format(outsegs[i + 1],
                                                                                        df.loc[outsegs[i + 1], 'maxElev'],
                                                                                        df.loc[outsegs[i + 1], 'minElev'],
                                                                                        minElev,
                                                                                        df.loc[outsegs[i + 1], 'minElev'],
                                                                                        oo))
                                df.loc[oo, 'maxElev'] = minElev
                        else:
                            break

                            # check again for zero-gradient lines
                dg.df = df
                comids0 = dg.check4zero_gradient(log=False)

                if len(comids0) > 0:
                    for c in comids0:
                        efp.write('{}\n'.format(c))
                    print("\nWarning!, the following comids had zero gradients:\n{}".format(comids0))
                    print("routing for these was turned off. Elevations must be fixed manually.\n" \
                        "See {}".format(self.error_reporting))
        return df

    def drop_crossing_lines(self, df):

        dg = self.dg
        crosses = dg.check4crossing_lines()

    def drop_duplicates(self, df):
        # loops or braids in NHD linework can result in duplicate lines after simplification
        # create column of line coordinates converted to strings
        df['ls_coords_str'] = [''.join(map(str, coords)) for coords in df.ls_coords]

        # identify duplicates; make common set of up and down comids for duplicates
        duplicates = np.unique(df.loc[df.duplicated('ls_coords_str'), 'ls_coords_str'])
        for dup in duplicates:
            alld = df[df.ls_coords_str == dup]
            upcomids = []
            dncomid = []
            for i, r in alld.iterrows():
                upcomids += r.upcomids
                dncomid += r.dncomid

            upcomids = list(set(upcomids).difference({0, '0'}))
            dncomid = list(set(dncomid).difference({0, '0'}))

            keep_comid = alld.index[0]
            df.at[keep_comid, 'upcomids'] = upcomids
            df.at[keep_comid, 'dncomid'] = dncomid
            #df.set_value(keep_comid, 'upcomids', upcomids)
            #df.set_value(keep_comid, 'dncomid', dncomid)
            for u in upcomids:
                df.at[u, 'dncomid'] = [keep_comid]
                #df.set_value(u, 'dncomid', [keep_comid])
            for d in dncomid:
                upids = set(df.loc[d, 'upcomids']).difference(set(alld.index[1:]))
                upids.add(alld.index[0])
                df.at[d, 'upcomids'] = list(upids)
                #df.set_value(d, 'upcomids', list(upids))
            df.drop(alld.index[1:], axis=0, inplace=True)

        # drop the duplicates (this may cause problems if multiple braids are routed to)
        #df = df.drop_duplicates('ls_coords_str') # drop rows from dataframe containing duplicates
        df = df.drop('ls_coords_str', axis=1)
        return df

    def setup_linesink_lakes(self, df):

        # read in elevations for NHD waterbodies (from preprocessing routine; needed for isolated lakes)
        wb_elevs = gisutils.shp2df(self.wb_centroids_w_elevations,
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
                df.loc[wb_comid, 'maxElev'] = wb_elevs[wb_comid]
                df.loc[wb_comid, 'minElev'] = wb_elevs[wb_comid] - 0.01
                df.loc[wb_comid, 'routing'] = 0
            else:
                df.loc[wb_comid, 'minElev'] = np.min(lines.minElev)
                df.loc[wb_comid, 'maxElev'] = np.min(lines.maxElev)

                # get upcomids and downcomid for lake,
                # by differencing all up/down comids for lines in lake, and comids in the lake
                upcomids = list(set([c for l in lines.upcomids for c in l]) - set(lines.index))
                dncomids = list(set([c for l in lines.dncomid for c in l]) - set(lines.index))

                # .at sets an entry for a single row/column location in a DataFrame
                # (to avoid confusion over specifying a single row but supplying a sequence)
                df.at[wb_comid, 'upcomids'] = upcomids
                df.at[wb_comid, 'dncomid'] = dncomids
                #df.set_value(wb_comid, 'upcomids', upcomids)
                #df.set_value(wb_comid, 'dncomid', dncomids)
                try:
                    assert all([True if isinstance(comid, list) else False for comid in df.upcomids])
                    assert all([True if isinstance(comid, list) else False for comid in df.dncomid])
                except:
                    j=2


                # make the lake the down-comid for the upcomids of the lake
                # (instead of the lines that represented the lake in the flowlines dataset)
                # do the same for the down-comid of the lake
                for u in [u for u in upcomids if int(u) > 0]:  # exclude outlets
                    df.at[u, 'dncomid'] = [wb_comid]
                    #df.set_value(u, 'dncomid', [wb_comid])
                for d in [d for d in dncomids if int(d) > 0]:
                    df.at[d, 'upcomids'] = [wb_comid]
                    #df.set_value(d, 'upcomids', [wb_comid])
                try:
                    assert all([True if isinstance(comid, list) else False for comid in df.upcomids])
                    assert all([True if isinstance(comid, list) else False for comid in df.dncomid])
                except:
                    j=2
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
                    df.loc[df.FTYPE != 'LakePond', 'dncomid'] = [[wb_comid if v == comid else v for v in l] for l in df[df.FTYPE != 'LakePond'].dncomid]
                    df.loc[df.FTYPE != 'LakePond', 'upcomids'] = [[wb_comid if v == comid else v for v in l] for l in df[df.FTYPE != 'LakePond'].upcomids]
                '''
                # get total length of lines representing lake (used later to estimate width)
                df.loc[wb_comid, 'total_line_length'] = np.sum(lines.LENGTHKM)

                # modifications to routed lakes
                #if df.loc[wb_comid, 'routing'] == 1:

                # enforce gradient in routed lakes; update elevations in downstream comids
                if df.loc[wb_comid, 'minElev'] == df.loc[wb_comid, 'maxElev']:
                    df.loc[wb_comid, 'minElev'] -= 0.01
                    dnids = df.loc[wb_comid, 'dncomid']
                    for dnid in [d for d in dnids if int(d) > 0]:
                        df.loc[dnid, 'maxElev'] -= 0.01

            #df['dncomid'] = [[d] if not isinstance(d, list) else d for d in df.dncomid]
            #df['upcomids'] = [[u] if not isinstance(u, list) else u for u in df.upcomids]
            # move begining/end coordinate of linear ring representing lake to outlet location (to ensure correct routing)
            # some routed lakes may not have an outlet
            # do this for both routed and unrouted (farfield) lakes, so that the outlet line won't cross the lake
            # (only tributaries are tested for crossing in step below)
            lake_coords = uniquelist(df.loc[wb_comid, 'ls_coords'])
            try:
                if len(df.loc[wb_comid, 'dncomid']) > 0 and int(dncomids[0]) != 0:
                    outlet_coords = df.loc[df.loc[wb_comid, 'dncomid'][0], 'ls_coords'][0]
                    closest_ind = closest_vertex_ind(outlet_coords, lake_coords)
                    lake_coords[closest_ind] = outlet_coords
                    next_ind = closest_ind + 1 if closest_ind < (len(lake_coords) - 1) else 0
                # for lakes without outlets, make the last coordinate the outlet so that it can be moved below
                else:
                    outlet_coords = lake_coords[-1]
                    next_ind = 0
            except:
                j=2

            inlet_coords = move_point_along_line(lake_coords[next_ind], outlet_coords, 1)

            new_coords = [inlet_coords] + lake_coords[next_ind:] + lake_coords[:next_ind]
            df.loc[wb_comid, 'ls_coords'] = [new_coords]

            # make sure inlets/outlets don't cross lines representing lake
            wb_geom = LineString(df.loc[wb_comid, 'ls_coords'])
            x = [c for c in upcomids if int(c) != 0 and LineString(df.loc[c, 'ls_coords']).crosses(wb_geom)]
            if len(x) > 0:
                for c in x:
                    ls_coords = list(df.loc[c, 'ls_coords'])  # want to copy, to avoid modifying df
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
                    df.at[c, 'ls_coords'] = ls_coords
                    #df.set_value(c, 'ls_coords', ls_coords)
            # drop the lines representing the lake from the lines dataframe
            df.drop(lines.index, axis=0, inplace=True)
        return df

    def list_updown_comids(self, df):

        farfield = df.COMID[df.farfield].tolist()
        # record up and downstream comids for lines
        lines = [l for l in df.index if l not in self.wblist and l not in farfield]
        #df['dncomid'] = len(df)*[[]]
        #df['upcomids'] = len(df)*[[]]
        #df.loc[lines, 'dncomid'] = [list(df[df['Hydroseq'] == df.loc[i, 'DnHydroseq']].index) for i in lines]
        #df.loc[lines, 'upcomids'] = [list(df[df['DnHydroseq'] == df.loc[i, 'Hydroseq']].index) for i in lines]
        df['upcomids'] = [[]] * len(df)
        df['dncomid'] = [[]] * len(df)
        dncomid, upcomids = [], []
        for l in lines:
            # set up/down comids that are not in the model domain to zero
            dncomid.append([d if d in lines else 0 for d in
                            list(df[df['Hydroseq'] == df.loc[l, 'DnHydroseq']].index)])
            upcomids.append([u if u in lines else 0 for u in
                             list(df[df['DnHydroseq'] == df.loc[l, 'Hydroseq']].index)])

        df.loc[lines, 'upcomids'] = np.array(upcomids, dtype=object)
        df.loc[lines, 'dncomid'] = np.array(dncomid, dtype=object)
        # enforce list datatype (apparently pandas flattens lists of lists len()=1 on assignment
        df['dncomid'] = [[d] if not isinstance(d, list) else d for d in df.dncomid]
        return df

    def make_linesinks(self, reuse_preprocessed_lines=False,
                       shp=None):

        with open(self.error_reporting, 'a') as efp:
            efp.write('\nMaking the lines...\n')

        if shp is None:
            shp = self.preprocessed_lines

        if reuse_preprocessed_lines:
            try:
                self.df = gisutils.shp2df(shp, index='COMID',
                                    index_dtype=self.int_dtype,
                                    true_values=['True'],
                                    false_values=['False'])
                self._enforce_dtypes(self.df)
            except:
                self.preprocess(save=True)
        else:
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

        self.wblist = set(df.loc[df.waterbody].index.values.astype(self.int_dtype)).difference({0})

        print('Assigning attributes for GFLOW input...')

        # routing
        # need to reformulate simplification so that "routed" means all routed LinesinkData,
        # not just those between the nearfield and farfield
        df['routing'] = (df.routed.values | df.nearfield.values).astype(int)

        # linesink elevations (lakes won't be populated yet)
        min_elev_col = [c for c in df.columns if 'minelev' in c.lower()][0]
        max_elev_col = [c for c in df.columns if 'maxelev' in c.lower()][0]
        df['minElev'] = df[min_elev_col] * self.zmult
        df['maxElev'] = df[max_elev_col] * self.zmult
        df['dStage'] = df['maxElev'] - df['minElev']

        # list upstream and downstream comids
        df = self.list_updown_comids(df)

        # discard duplicate LinesinkData that result from braids in NHD and line simplification
        df = self.drop_duplicates(df)

        # method to represent lakes with LinesinkData
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

        # fix LinesinkData where max and min elevations are the same
        df = self.adjust_zero_gradient(df)

        # end streams
        # evaluate whether downstream segment is in farfield
        downstream_ff = []
        for i in range(len(df)):
            try:
                dff = df.loc[df.iloc[i].dncomid[0], 'farfield'].item()
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
            df.loc[df['FTYPE'] == 'LakePond', 'width'] = \
                np.vectorize(lake_width)(df.loc[df['FTYPE'] == 'LakePond', 'AREASQKM'],
                                         df.loc[df['FTYPE'] == 'LakePond', 'total_line_length'], self.lmbda)

        # resistance
        df['resistance'] = self.resistance
        df.loc[df['farfield'], 'resistance'] = 0

        # depth
        df['depth'] = self.global_streambed_thickness

        # resistance parameter (scenario)
        df['ScenResistance'] = self.ScenResistance
        df.loc[df['farfield'], 'ScenResistance'] = '__NONE__'

        # check box for "include in Scenario" in the GUI
        df['chkScenario'] = self.chkScenario

        # linesink location
        df.loc[df['FTYPE'] != 'LakePond', 'AutoSWIZC'] = 1  # Along stream centerline
        df.loc[df['FTYPE'] == 'LakePond', 'AutoSWIZC'] = 2  # Along surface water boundary

        # additional check to drop isolated lines
        isolated = [c for c in df.index if len(df.loc[c].dncomid) == 0 and len(df.loc[c].upcomids) == 0
                    and c not in self.wblist]
        #df = df.drop(isolated, axis=0)

        # drop lines below minimum length that are not Lakes
        if self.farfield_length_threshold > 0.:
            criteria = ~df.farfield.values | (df.FTYPE == 'LakePond') |\
                       (df.farfield.values & (df.LENGTHKM >= self.farfield_length_threshold))
            df = df.loc[criteria].copy()

        if self.routed_area_length_threshold > 0.:
            criteria = ~df.routed.values | (df.FTYPE == 'LakePond') |\
                       (df.routed.values & (df.LENGTHKM >= self.routed_area_length_threshold))
            df = df.loc[criteria].copy()


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

        # run diagnostics on lines and report errors
        self.run_diagnostics()

        # write shapefile of results
        # convert lists in dn and upcomid columns to strings (for writing to shp)
        # clean up any dtypes that were altered from sloppy use of pandas
        self._enforce_dtypes(df)
        self.df = df
        self.write_shapefile()
        print('Done!')

    def map_confluences(self):

        upsegs = self.df.upcomids.tolist()
        maxsegs = np.array([np.max([int(uu) for uu in u]) if len(u) > 0
                            else 0 for u in upsegs])
        seglengths = np.array([len(u) for u in upsegs])
        # setup dataframe of confluences
        # confluences are where segments have upsegs (no upsegs means the reach 1 is a headwater)
        confluences = self.df.loc[(seglengths > 0) & (maxsegs > 0), ['COMID', 'upcomids']].copy()

        confluences['elev'] = [0] * len(confluences)
        nconfluences = len(confluences)
        print('Mapping {} confluences and updating segment min/max elevations...'.format(nconfluences))
        for i, r in confluences.iterrows():

            # confluence elevation is the minimum of the ending segments minimums, starting segments maximums
            endsmin = np.min(self.df.loc[self.df.COMID.isin(r.upcomids), 'minElev'].values)
            startmax = np.max(self.df.loc[self.df.COMID == i, 'maxElev'].values)
            cfelev = np.min([endsmin, startmax])
            confluences.loc[i, 'elev'] = cfelev

            upcomids = [u for u in r.upcomids if int(u) > 0]
            if len(upcomids) > 0:
                self.df.loc[upcomids, 'minElev'] = cfelev
            self.df.loc[i, 'maxElev'] = cfelev

        self.confluences = confluences
        self.df['dStage'] = self.df['maxElev'] - self.df['minElev']
        print('Done, see confluences attribute.')

    def run_diagnostics(self):
        """Run the diagnostic suite on the LinesinkData instance."""
        dg = self.dg
        dg.check_vertices()
        dg.check4crossing_lines()
        dg.check4zero_gradient()

    def map_outsegs(self):
        '''
        from Mat2, returns dataframe of all downstream segments (will not work with circular routing!)
        '''
        outsegsmap = pd.DataFrame(self.df.COMID)
        outsegs = pd.Series([d[0] if len(d) > 0 else 0 for d in self.df.dncomid], index=self.df.index)
        max_outseg = outsegsmap[outsegsmap.columns[-1]].astype(int).max()
        knt = 2
        while max_outseg > 0:
            outsegsmap['outseg{}'.format(knt)] = [outsegs[s] if int(s) > 0 else 0
                                                  for s in outsegsmap[outsegsmap.columns[-1]]]
            max_outseg = outsegsmap[outsegsmap.columns[-1]].astype(int).max()
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
                comids.append(r.Label.split()[0].strip())
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
        HUCs_df = gisutils.shp2df(self.HUC_shp, index=self.HUC_name_field)
        df[self.HUC_name_field] = len(df) * [None]
        for HUC in HUCs_df.index:
            lines = [line.intersects(HUCs_df.loc[HUC, 'geometry']) for line in df['geometry']]
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

        df['chkScenario'] = [s.lower() for s in df.chkScenario.astype(str)]

        nlines = sum([len(p) - 1 for p in df.ls_coords])

        print('writing {} lines to {}'.format(nlines, outfile))
        ofp = open(outfile, 'w')
        ofp.write('<?xml version="1.0"?>\n')
        ofp.write('<LinesinkStringFile version="1">\n')
        ofp.write('\t<ComputationalUnits>{}</ComputationalUnits>\n'
                  '\t<BasemapUnits>{}</BasemapUnits>\n\n'.format(self.ComputationalUnits.capitalize(),
                                                                 self.BasemapUnits.capitalize()))

        for comid in df.index:
            ofp.write('\t<LinesinkString>\n')
            ofp.write('\t\t<Label>{}</Label>\n'.format(df.loc[comid, 'ls_name']))
            ofp.write('\t\t<HeadSpecified>1</HeadSpecified>\n')
            ofp.write('\t\t<StartingHead>{:.2f}</StartingHead>\n'.format(df.loc[comid, 'maxElev']))
            ofp.write('\t\t<EndingHead>{:.2f}</EndingHead>\n'.format(df.loc[comid, 'minElev']))
            ofp.write('\t\t<Resistance>{}</Resistance>\n'.format(df.loc[comid, 'resistance']))
            ofp.write('\t\t<Width>{:.2f}</Width>\n'.format(df.loc[comid, 'width']))
            ofp.write('\t\t<Depth>{:.2f}</Depth>\n'.format(df.loc[comid, 'depth']))
            ofp.write('\t\t<Routing>{}</Routing>\n'.format(df.loc[comid, 'routing']))
            ofp.write('\t\t<EndStream>{}</EndStream>\n'.format(df.loc[comid, 'end_stream']))
            ofp.write('\t\t<OverlandFlow>0</OverlandFlow>\n')
            ofp.write('\t\t<EndInflow>0</EndInflow>\n')
            ofp.write('\t\t<ScenResistance>{}</ScenResistance>\n'.format(df.loc[comid, 'ScenResistance']))
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
            ofp.write('\t\t<Farfield>{:.0f}</Farfield>\n'.format(df.loc[comid, 'farfield']))
            ofp.write('\t\t<chkScenario>{}</chkScenario>\n'.format(df.loc[comid, 'chkScenario'])) # include linesink in PEST 'scenarios'
            ofp.write('\t\t<AutoSWIZC>{:.0f}</AutoSWIZC>\n'.format(df.loc[comid, 'AutoSWIZC']))
            ofp.write('\t\t<DefaultResistance>{:.2f}</DefaultResistance>\n'.format(df.loc[comid, 'resistance']))
            ofp.write('\t\t<Vertices>\n')

            # now write out linesink vertices
            for x, y in df.loc[comid, 'ls_coords']:
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

        gisutils.df2shp(df, outfile, crs=self.crs)


class linesinks(LinesinkData):
    def __init__(self, *args, **kwargs):
        warnings.warn("The 'linesinks' class was renamed to LinesinkData to better follow pep 8.",
                      DeprecationWarning)
        LinesinkData.__init__(self, *args, **kwargs)


class InputFileMissing(Exception):
    def __init__(self, infile):
        self.infile = infile

    def __str__(self):
        return ('\n\nCould not open or parse input file {0}.\nCheck for errors in XML formatting.'.format(self.infile))


class EmptyDataFrame(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return ('\n\nEmpty DataFrame!')