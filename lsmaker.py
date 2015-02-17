__author__ = 'aleaf'

import xml.etree.ElementTree as ET
import numpy as np
import os
import GISio
from shapely.geometry import Polygon, LineString
from shapely.ops import cascaded_union
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


### Functions #############################

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
    w = 2 * w_parameter(estwidth, lmbda) # assumes stream is rep. by single linesink
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
    return w # feet


def name(x):
    """Abbreviations for naming linesinks from names in NHDPlus
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
        for k, v in abb.iteritems():
            name = name.replace(k, v)
    else:
        name = '{} unnamed'.format(x.name)
    return name


def write_lss(df, outfile):
    """write GFLOW linesink XML (lss) file from dataframe df
    """
    # global inputs
    depth = 3 # streambed thickness
    DefaultResistance = 0.3
    ComputationalUnits = 'Feet' # 'Feet' or 'Meters'; for XML output file
    BasemapUnits = 'Meters'

    nlines = sum([len(p)-1 for p in df.ls_coords])

    print 'writing {} lines to {}'.format(nlines, outfile)
    ofp = open(outfile,'w')
    ofp.write('<?xml version="1.0"?>\n')
    ofp.write('<LinesinkStringFile version="1">\n')
    ofp.write('\t<ComputationalUnits>{}</ComputationalUnits>\n\t<BasemapUnits>{}</BasemapUnits>\n\n'.format(ComputationalUnits, BasemapUnits))

    for comid in df.index:
        ofp.write('\t<LinesinkString>\n')
        ofp.write('\t\t<Label>{}</Label>\n'.format(df.ix[comid, 'ls_name']))
        ofp.write('\t\t<HeadSpecified>1</HeadSpecified>\n')
        ofp.write('\t\t<StartingHead>{:.2f}</StartingHead>\n'.format(df.ix[comid, 'maxElev']))
        ofp.write('\t\t<EndingHead>{:.2f}</EndingHead>\n'.format(df.ix[comid, 'minElev']))
        ofp.write('\t\t<Resistance>{}</Resistance>\n'.format(df.ix[comid, 'resistance']))
        ofp.write('\t\t<Width>{:.2f}</Width>\n'.format(df.ix[comid, 'width']))
        ofp.write('\t\t<Depth>{:.2f}</Depth>\n'.format(resistance))
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
        ofp.write('\t\t<chkScenario>true</chkScenario>\n') # include linesink in PEST 'scenarios'
        ofp.write('\t\t<AutoSWIZC>{:.0f}</AutoSWIZC>\n'.format(df.ix[comid, 'AutoSWIZC']))
        ofp.write('\t\t<DefaultResistance>{:.2f}</DefaultResistance>\n'.format(DefaultResistance))
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


def closest_vertex(point, shape):
    """Returns index of closest vertex in shapely geometry object
    """
    X, Y = np.ravel(shape.coords.xy[0]), np.ravel(shape.coords.xy[1])
    dX, dY = X - point[0], Y - point[1]
    closest_ind = np.argmin(np.sqrt(dX**2 + dY**2))
    return closest_ind


class linesinks:

    def __init__(self, infile):

        try:
            inpardat = ET.parse(infile)
        except:
            raise(InputFileMissing(infile))

        inpars = inpardat.getroot()

        # setup the working directory (default to current directory)
        try:
            self.path = inpars.findall('.//working_dir')[0].text
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        except:
            self.path = os.getcwd()

        # global settings
        self.preproc = self.tf2flag(inpars.findall('.//preprocess')[0].text)
        self.z_mult = float(inpars.findall('.//zmult')[0].text) # elevation units multiplier (from NHDPlus cm to model units)
        self.resistance = float(inpars.findall('.//resistance')[0].text) # (days); c in documentation
        self.H = float(inpars.findall('.//H')[0].text) # aquifer thickness in model units
        self.k = float(inpars.findall('.//k')[0].text) # hydraulic conductivity of the aquifer in model units
        self.lmbda = np.sqrt(10 * 100 * 0.3)
        self.ScenResistance = inpars.findall('.//ScenResistance')[0].text

        # model domain
        self.farfield = inpars.findall('.//farfield')[0].text
        self.nearfield = inpars.findall('.//nearfield')[0].text
        self.split_by_HUC = self.tf2flag(inpars.findall('.//split_by_HUC')[0].text)
        self.HUC_shp = inpars.findall('.//HUC_shp')[0].text
        self.HUC_name_field = inpars.findall('.//HUC_name_field')[0].text

        # simplification
        self.nearfield_tolerance = float(inpars.findall('.//nearfield_tolerance')[0].text)
        self.farfield_tolerance = float(inpars.findall('.//farfield_tolerance')[0].text)
        self.min_farfield_order = int(inpars.findall('.//min_farfield_order')[0].text)
        self.min_waterbody_size = float(inpars.findall('.//min_waterbody_size')[0].text)

        # NHD files
        self.flowlines = inpars.findall('.//flowlines')[0].text
        self.elevslope = inpars.findall('.//elevslope')[0].text
        self.PlusFlowVAA = inpars.findall('.//PlusFlowVAA')[0].text
        self.waterbodies = inpars.findall('.//waterbodies')[0].text
        self.prj = inpars.findall('.//prj')[0].text

        # preprocessed files
        self.preprocdir = ''
        self.DEM = inpars.findall('.//DEM')[0].text
        self.elevs_field = inpars.findall('.//elevs_field')[0].text
        self.DEM_zmult = float(inpars.findall('.//DEM_zmult')[0].text)

        try:
            self.flowlines_clipped = inpars.findall('.//flowlines_clipped')[0].text
            self.preprocdir = os.path.split(self.flowlines_clipped)[0]
        except:
            self.flowlines_clipped = os.path.join(self.path, self.preprocdir, 'flowlines_clipped.shp')
        try:
            self.waterbodies_clipped = inpars.findall('.//waterbodies_clipped')[0].text
        except:
            self.waterbodies_clipped = os.path.join(self.path, self.preprocdir, 'waterbodies_clipped.shp')
        try:
            self.farfield_mp = inpars.findall('.//farfield_multipolygon')[0].text
        except:
            self.farfield_mp = os.path.join(self.path, self.preprocdir, 'ff_cutout.shp')

        self.wb_centroids_w_elevations = self.waterbodies_clipped[:-4] + '_points.shp' # elevations extracted during preprocessing routine
        self.elevs_field = 'DEM_m' # field in wb_centroids_w_elevations containing elevations

        # outputs
        self.outfile_basename = inpars.findall('.//outfile_basename')[0].text
        self.error_reporting = inpars.findall('.//error_reporting')[0].text
        self.efp = open(self.error_reporting, 'w')

    def tf2flag(self, intxt):
        # converts text written in XML file to True or False flag
        if intxt.lower() == 'true':
            return True
        else:
            return False

    def preprocess_arcpy(self):
        '''
        requires arcpy
        incomplete; does not include projection
        '''
        import arcpy
        # initialize the arcpy environment
        arcpy.env.workspace = self.path
        arcpy.env.overwriteOutput = True
        arcpy.env.qualifiedFieldNames = False
        arcpy.CheckOutExtension("spatial") # Check spatial analyst license

        # make the output directory if it doesn't exist yet
        if len(self.preprocdir) > 0 and not os.path.isdir(self.preprocdir):
            os.makedirs(self.preprocdir)

        print 'clipping {} and {} to {}...'.format(self.flowlines, self.waterbodies, self.farfield)
        arcpy.Clip_analysis(self.flowlines, self.farfield, self.flowlines_clipped)
        arcpy.Clip_analysis(self.waterbodies, self.farfield, self.waterbodies_clipped)
        print 'clipped flowlines written to {}; clipped waterbodies written to {}'\
            .format(self.flowlines_clipped, self.waterbodies_clipped)

        print '\nremoving interior from farfield polygon...'
        arcpy.Erase_analysis(self.farfield, self.nearfield, self.farfield_mp)
        print 'farfield donut written to {}'.format(self.farfield_mp)

        print '\ngetting NHD Waterbody elevations from DEM (needed for isolated lakes)'
        arcpy.FeatureToPoint_management(self.waterbodies_clipped, self.wb_centroids_w_elevations)
        arcpy.sa.ExtractMultiValuesToPoints(self.wb_centroids_w_elevations, [[self.DEM, self.elevs_field]])
        print 'waterbody elevations written to point dataset {}'.format(self.wb_centroids_w_elevations)
        print '\nDone.'

    def preprocess(self, save=True):
        '''
        associate NHD tabular information to linework
        intersect lines with nearfield and farfield domains
        edit linework
            - remove farfield streams lower than minimum order
            - remove lakes smaller than minimum size
            - convert lakes from polygons to lines; merge with lines
        '''

        # open error reporting file
        self.efp = open(self.error_reporting, 'a')
        self.efp.write('\nPreprocessing...\n')

        print '\nAssembling input...'
        # read linework shapefile into pandas dataframe
        df = GISio.shp2df(self.flowlines_clipped, geometry=True, index='COMID')
        elevs = GISio.shp2df(self.elevslope, index='COMID', clipto=df)
        pfvaa = GISio.shp2df(self.PlusFlowVAA, index='COMID', clipto=df)
        wbs = GISio.shp2df(self.waterbodies_clipped, index='COMID', geometry=True)

        # check for MultiLineStrings / MultiPolygons and drop them (these are features that were fragmented by the boundaries)
        mls = [i for i in df.index if 'multi' in df.ix[i]['geometry'].type.lower()]
        df = df.drop(mls, axis=0)
        # get multipolygons using iterator; for some reason the above approach didn't work with the wbs dataframe
        mpoly_inds = [i for i, t in enumerate(wbs['geometry']) if 'multi' in t.type.lower()]
        wbs = wbs.drop(wbs.index[mpoly_inds], axis=0)

        # join NHD tables to lines
        lsuffix = 'fl'
        df = df.join(elevs, how='inner', lsuffix=lsuffix, rsuffix='elevs')
        df = df.join(pfvaa, how='inner', lsuffix=lsuffix, rsuffix='pfvaa')

        # read in nearfield and farfield boundaries
        nf = GISio.shp2df(self.nearfield, geometry=True)
        nfg = nf.iloc[0]['geometry'] # polygon representing nearfield
        ff = GISio.shp2df(os.path.join(self.path, 'ff_cutout.shp'), geometry=True)
        ffg = ff.iloc[0]['geometry'] # shapely geometry object for farfield (polygon with interior ring for nearfield)

        print '\nidentifying farfield and nearfield linesinks...'
        df['farfield'] = [line.intersects(ffg) and not line.intersects(nfg) for line in df['geometry']]
        wbs['farfield'] = [poly.intersects(ffg) for poly in wbs['geometry']]

        print 'removing farfield streams lower than {} order...'.format(self.min_farfield_order)
        df = df.drop(df.index[np.where(df['farfield'] & (df['StreamOrde'] < self.min_farfield_order))], axis=0)

        print 'dropping waterbodies that are not lakes larger than {}...'.format(self.min_waterbody_size)
        wbs = wbs.drop(wbs.index[np.where((wbs['AREASQKM'] < self.min_waterbody_size) | (wbs['FTYPE'] != 'LakePond'))], axis=0)

        print 'merging waterbodies with coincident boundaries...'
        dropped = []
        for wb_comid in wbs.index:

            # skipped already merged
            if wb_comid in dropped:
                continue

            overlapping = wbs.ix[[wbs.ix[wb_comid, 'geometry'].intersects(r) \
                                                                for r in wbs.geometry]]
            basering_comid = overlapping.sort('FTYPE').index[0] # sort to prioritize features with names
            # two or more shapes in overlapping signifies a coincident boundary
            if len(overlapping > 1):
                merged = cascaded_union([r for r in overlapping.geometry]).exterior
                wbs.ix[basering_comid, 'geometry'] = Polygon(merged) # convert from linear ring back to polygon (for next step)

                todrop = [wbc for wbc in overlapping.index if wbc != basering_comid]
                dropped += todrop
                wbs = wbs.drop(todrop) # only keep merged feature
                # replace references to dropped waterbody in lines
                for wbc in overlapping.index:
                    df.ix[df['WBAREACOMI'] == wbc, 'WBAREACOMI'] = basering_comid
                    # df['WBAREACOMI'] = [basering_comid if c == wbc else c for df['WBAREACOMI']]

        # swap out polygons in lake geometry column with the linear rings that make up their exteriors
        print 'converting lake exterior polygons to lines...'
        wbs['geometry'] = wbs['geometry'].apply(lambda x: x.exterior)
        wbs['geometry'] = [LineString(g) for g in wbs.geometry]
        wbs['waterbody'] = [True] * len(wbs)

        print 'merging flowline and waterbody datasets...'
        df['waterbody'] = [False] * len(df)
        df = df.append(wbs)
        df.COMID = df.index

        print 'Done with preprocessing.'
        if save:
            GISio.df2shp(df, 'lines.shp', prj=self.prj)

        self.df = df

    def simplify_lines(self, nearfield_tolerance=None, farfield_tolerance=None):

        if nearfield_tolerance is None:
            nearfield_tolerance = self.nearfield_tolerance
            farfield_tolerance = self.farfield_tolerance

        if isinstance(self.df.farfield.iloc[0], basestring):
            self.df.loc[:, 'farfield'] = [True if f.lower() == 'true' else False for f in self.df.farfield]

        print 'simplifying NHD linework geometries...'
        # simplify line and waterbody geometries
        #(see http://toblerity.org/shapely/manual.html)
        '''
        df['geometry_nf'] = df['geometry'].map(lambda x: x.simplify(self.nearfield_tolerance))
        df['geometry_ff'] = df['geometry'].map(lambda x: x.simplify(self.farfield_tolerance))
        '''
        df = self.df[['farfield', 'geometry']]

        ls_geom = np.array([LineString()] * len(df))
        domain_tol = [nearfield_tolerance, farfield_tolerance]
        for i, domain in enumerate([np.invert(df.farfield).values, df.farfield.values]):

            # simplify the linesinks in the domain; add simplified geometries to global geometry column
            # assign geometries to numpy array first and then to df (had trouble assigning with pandas)
            ls_geom[domain] = [g.simplify(domain_tol[i]) for g in df.ix[domain, 'geometry'].tolist()]

        df.loc[:, 'ls_geom'] = ls_geom

        # convert geometries to coordinates
        def xy_coords(x):
            xy = zip(x.xy[0], x.xy[1])
            return xy

        # add column of lists, containing linesink coordinates
        df.loc[:, 'ls_coords'] = df.ls_geom.apply(xy_coords)

        return df

    def prototype(self, nftol=[10, 50, 100, 200, 500], fftol=500):

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
            GISio.df2shp(df, outshp, geo_column='ls_geom', prj=self.prj)

        plt.figure()
        plt.plot(nftol, nlines)
        plt.xlabel('Distance tolerance')
        plt.ylabel('Number of lines')
        plt.savefig(self.outfile_basename + 'tol_vs_nlines.pdf')

    def makeLineSinks(self, shp=None):
        self.efp = open(self.error_reporting, 'a')
        self.efp.write('\nMaking the lines...\n')

        if shp:
            self.df = GISio.shp2df(shp, index='COMID', geometry=True, true_values=['True'], false_values=['False'])

        df = self.df

        self.lines_df = self.simplify_lines()

        df['ls_geom'] = self.lines_df['ls_geom']
        df['ls_coords'] = self.lines_df['ls_coords']

        self.wblist = df.ix[df.waterbody].index

        print 'Assigning attributes for GFLOW input...'


        '''
        df.loc[np.invert(df.farfield), 'ls_coords'] = df.ix[np.invert(df.farfield), 'geometry_nf'].apply(xy_coords) # nearfield coordinates
        df.loc[df.farfield, 'ls_coords'] = df.ix[df.farfield, 'geometry_ff'].apply(xy_coords) # farfield coordinates
        '''

        # routing
        df['routing'] = len(df)*[1]
        df.loc[df['farfield'], 'routing'] = 0 # turn off all routing in farfield (conversely, nearfield is all routed)


        # linesink elevations (lakes won't be populated yet)
        min_elev_col = [c for c in df.columns if 'minelev' in c.lower()][0]
        max_elev_col = [c for c in df.columns if 'maxelev' in c.lower()][0]
        df['minElev'] = df[min_elev_col] * self.z_mult
        df['maxElev'] = df[max_elev_col] * self.z_mult
        df['dStage'] = df['maxElev'] - df['minElev']


        # record up and downstream comids for lines
        lines = [l for l in df.index if l not in self.wblist]
        df['dncomid'] = len(df)*[[]]
        df['upcomids'] = len(df)*[[]]
        df.ix[lines, 'dncomid'] = [list(df[df['Hydroseq'] == df.ix[i, 'DnHydroseq']].index) for i in lines]
        df.ix[lines, 'upcomids'] = [list(df[df['DnHydroseq'] == df.ix[i, 'Hydroseq']].index) for i in lines]


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
                if i == 6820078:
                    j=2
                upcomids += r.upcomids
                dncomid += r.dncomid

                # if nothing routes to the braid, drop it (keep the duplicate braid that is routed too)
                if len(r.upcomids) == 0:
                    df.drop(i, axis=0)

            upcomids, dncomid = list(set(upcomids)), list(set(dncomid))

            alld = df[df.ls_coords_str == dup]
            for i, r in alld.iterrows():
                df.set_value(i, 'upcomids', upcomids)
                df.set_value(i, 'dncomid', dncomid)

        # drop the duplicates (this may cause problems if multiple braids are routed to)
        df = df.drop_duplicates('ls_coords_str') # drop rows from dataframe containing duplicates
        df = df.drop('ls_coords_str', axis=1)


        # read in elevations for NHD waterbodies (from preprocessing routine; needed for isolated lakes)
        wb_elevs = GISio.shp2df(self.wb_centroids_w_elevations, index='COMID')
        wb_elevs = wb_elevs[self.elevs_field] * self.DEM_zmult

        # identify lines that represent lakes
        # get elevations, up/downcomids, and total lengths for those lines
        # assign attributes to lakes, then drop the lines

        for wb_comid in self.wblist:

            lines = df[df['WBAREACOMI'] == wb_comid]
            if wb_comid == 9022741:
                j=2

            # isolated lakes have no overlapping lines and no routing
            if len(lines) == 0:
                df.ix[wb_comid, 'maxElev'] = wb_elevs[wb_comid]
                df.ix[wb_comid, 'minElev'] = wb_elevs[wb_comid] - 0.01
                df.ix[wb_comid, 'routing'] = 0
                continue
            # get upcomids and downcomid for lake,
            # by differencing all up/down comids for lines in lake, and comids in the lake

            #df.ix[wb_comid, 'upcomids'] = list(set([c for l in lines.upcomids for c in l]) - set(lines.index))
            #df.ix[wb_comid, 'dncomid'] = list(set([c for l in lines.dncomid for c in l]) - set(lines.index))
            df.set_value(wb_comid, 'upcomids', list(set([c for l in lines.upcomids for c in l]) - set(lines.index)))
            df.set_value(wb_comid, 'dncomid', list(set([c for l in lines.dncomid for c in l]) - set(lines.index)))

            df.ix[wb_comid, 'minElev'] = np.min(lines.minElev)
            df.ix[wb_comid, 'maxElev'] = np.min(lines.maxElev)

            # update all up/dn comids in lines dataframe that reference the lines inside of the lakes
            # (replace those references with the comids for the lakes)
            for comid in lines.index:
                df.ix[df.FTYPE != 'LakePond', 'dncomid'] = [[wb_comid if v == comid else v for v in l] for l in df[df.FTYPE != 'LakePond'].dncomid]
                df.ix[df.FTYPE != 'LakePond', 'upcomids'] = [[wb_comid if v == comid else v for v in l] for l in df[df.FTYPE != 'LakePond'].upcomids]

            # get total length of lines representing lake (used later to estimate width)
            df.ix[wb_comid, 'total_line_length'] = np.sum(lines.LengthKM)

            # modifications to routed lakes
            if df.ix[wb_comid, 'routing'] == 1:

                # enforce gradient; update elevations in downstream comids
                if df.ix[wb_comid, 'minElev'] == df.ix[wb_comid, 'maxElev']:
                    df.ix[wb_comid, 'minElev'] -= 0.01
                    for dnid in df.ix[wb_comid, 'dncomid']:
                        df.ix[dnid, 'maxElev'] -= 0.01

                # move begining/end coordinate of linear ring representing lake to outlet location (to ensure correct routing)
                # some routed lakes may not have an outlet
                if len(df.ix[wb_comid, 'dncomid']) > 0:
                    outlet_coords = df.ix[df.ix[wb_comid, 'dncomid'][0], 'ls_coords'][0]

                    #closest_ind = self.closest_vertex(outlet_coords, df.ix[wb_comid, 'geometry_nf'])
                    closest_ind = closest_vertex(outlet_coords, df.ix[wb_comid, 'ls_geom'])
                    '''
                    X, Y = np.ravel(df.ix[wb_comid, 'geometry_nf'].coords.xy[0]), np.ravel(df.ix[wb_comid, 'geometry_nf'].coords.xy[1])
                    dX, dY = X - outlet_coords[0], Y - outlet_coords[1]
                    closest_ind = np.argmin(np.sqrt(dX**2 + dY**2))
                    '''
                    # make new set of vertices that start and end at outlet location (and only include one instance of previous start/end!)
                    new_coords = [outlet_coords] + df.ix[wb_comid, 'ls_coords'][closest_ind+1:] + \
                                 df.ix[wb_comid, 'ls_coords'][1:closest_ind] + [outlet_coords]
                    df.set_value(wb_comid, 'ls_coords', new_coords)

            # drop the lines representing the lake from the lines dataframe
            df = df.drop(lines.index)

        print '\nmerging or splitting lines with only two vertices...'
        # find all routed comids with only 1 line; merge with neighboring comids
        # (GFLOW requires two lines for routed streams)

        def bisect(coords):
            # add vertex to middle of single line segment
            coords = np.array(coords)
            mid = 0.5 * (coords[0] + coords[-1])
            new_coords = map(tuple, [coords[0], mid, coords[-1]])
            return new_coords

        df['nlines'] = [len(coords) for coords in df.ls_coords]
        #comids1 = list(df[(df['nlines'] < 3) & (df['routing'] == 1)]['COMID'+lsuffix])
        comids1 = list(df[(df['nlines'] < 3) & (df['routing'] == 1)].index)
        self.efp.write('\nunrouted comids of length 1 that were dropped:\n')
        for comid in comids1:


            # get up and down comids/elevations; only consider upcomid/downcomids that are streams (exclude lakes)
            #upcomids = [c for c in df[df.index == comid]['upcomids'].item() if c not in self.wblist]
            #dncomid = [c for c in df[df.index == comid]['dncomid'].item() if c not in self.wblist]
            upcomids = [c for c in df[df.index == comid]['upcomids'].item()] # allow lakes and lines to be merged (if their vertices coincide)
            dncomid = [c for c in df[df.index == comid]['dncomid'].item()]
            merged = False
            if comid == 13396559 or comid == 13396569 or comid == 13396555 or comid == 13397241:
                j=2
            if comid == 6820088:
                j=2

            try:
                # first try to merge with downstream comid
                if len(dncomid) > 0:
                    # only merge if start of downstream comid coincides with last line segment
                    if df.ix[comid].ls_coords[-1] == df.ix[dncomid[0]].ls_coords[0]:
                        new_coords = df.ix[comid].ls_coords + df.ix[dncomid[0]].ls_coords[1:]
                        df.set_value(dncomid[0], 'ls_coords', new_coords) # update coordinates in dncomid
                        df.loc[dncomid[0], 'maxElev'] = df.ix[comid].maxElev # update max elevation

                        df = df.drop(comid, axis=0)

                        # record merged comid and replace references to it (as a dncomid)
                        replacement = dncomid[0]
                        df['dncomid'] = [[replacement if v == comid else v for v in l] for l in df['dncomid']]

                        # add upcomids of merged segment to its replacement
                        new_upcomids = list(set(df.ix[replacement, 'upcomids'] + upcomids))
                        new_upcomids.remove(comid)
                        df.set_value(replacement, 'upcomids', new_upcomids)

                        #merged = True
                    else: # split it
                        new_coords = bisect(df.ix[comid].ls_coords)
                        df.set_value(comid, 'ls_coords', new_coords)

                elif len(upcomids) > 0: # merge into first upstream comid; then drop
                    for uid in upcomids:
                        # check if upstream end coincides with current start
                        if df.ix[uid].ls_coords[-1] == df.ix[comid].ls_coords[0]:
                            new_coords = df.ix[uid].ls_coords + df.ix[comid].ls_coords[1:]
                            df.set_value(uid, 'ls_coords', new_coords) # update coordinates in upcomid
                            df.loc[uid, 'minElev'] = df.ix[comid].minElev # update min elevation

                            df = df.drop(comid, axis=0)

                            # record merged comid and replace references to it (in the upcomids list of the downstream comid)
                            replacement = uid

                            new_upcomids = df.ix[dncomid[0], 'upcomids'].replace(comid, replacement)
                            df.set_value(replacement, 'upcomids', new_upcomids)

                            # update the dncomids of the replacement with those of the merged comid
                            df.set_value(replacement, 'dncomids', dncomid[0])

                            merged = True
                            break
                        else: # split it (for Nicolet, no linesinks were in this category)
                            continue
                    if not merged:
                        new_coords = bisect(df.ix[comid].ls_coords)
                        df.set_value(comid, 'ls_coords', new_coords)

                else: # the segment is not routed to any up/dn comids that aren't lakes
                    # split it for now (don't want to drop it if it connects to a lake)
                    new_coords = bisect(df.ix[comid].ls_coords)
                    df.set_value(comid, 'ls_coords', new_coords)
            except:
                pass
        '''
            if merged:
                # update any references to current comid (clunkly because each row is a list)
                df['dncomid'] = [[replacement if v == comid else v for v in l] for l in df['dncomid']]
                df['upcomids'] = [[replacement if v == comid else v for v in l] for l in df['upcomids']]
        '''

        print "adjusting elevations for comids with zero-gradient..."

        comids0 = list(df[df['dStage'] == 0].index)
        self.efp.write('\nzero-gradient errors:\n')
        self.efp.write('comid, upcomids, downcomid, elevmax, elevmin\n')
        zerogradient = []

        for comid in comids0:

            # get up and down comids/elevations
            upcomids = df[df.index == comid]['upcomids'].item()
            upelevsmax = [df[df.index == uid]['maxElev'].item() for uid in upcomids]
            dncomid = df[df.index == comid]['dncomid'].item()
            dnelevmin = [df[df.index == dnid]['minElev'].item() for dnid in dncomid]

            # adjust elevations for zero gradient comid if there is room
            if len(upcomids) == 0:
                df.loc[comid, 'maxElev'] += 0.01
            elif len(dncomid) == 0:
                df.loc[comid, 'minElev'] -= 0.01
            elif len(upcomids) > 0 and np.min(upelevsmax) > df.ix[comid, 'maxElev']:
                df.loc[comid, 'maxElev'] = 0.5 * (df.ix[comid, 'maxElev'] + np.min(upelevsmax))
            elif len(dncomid) > 0 and dnelevmin < df.ix[comid, 'minElev']:
                df.loc[comid, 'minElev'] = 0.5 * (df.ix[comid, 'minElev'] + dnelevmin)

            # otherwise, downstream and upstream comids are also zero gradient; report to error file
            else:
                farfield = df.ix[comid, 'farfield']
                if not farfield:
                    self.efp.write('{},{},{},{:.2f},{:.2f}\n'.format(comid, upcomids, dncomid, df.ix[comid, 'maxElev'].item(),
                              df.ix[comid, 'minElev'].item()))
                    #df.loc[comid, 'routing'] = 0
                    #just increase the max elev slightly to get around zero-gradient error
                    #df.loc[comid, 'maxElev'] += 0.01
                    zerogradient.append(comid)

        print "\nWarning!, the following comids had zero gradients:\n{}".format(zerogradient)
        print "routing for these was turned off. Elevations must be fixed manually"


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
        df.loc[downstream_ff, 'end_stream'] = 1 # set


        # widths for lines
        arbolate_sum_col = [c for c in df.columns if 'arbolate' in c.lower()][0]
        df['width'] = df[arbolate_sum_col].map(lambda x: width_from_arboate(x, self.lmbda))

        # widths for lakes
        df.ix[df['FTYPE'] == 'LakePond', 'width'] = \
            np.vectorize(self.lake_width)(df.ix[df['FTYPE'] == 'LakePond', 'AREASQKM'], df.ix[df['FTYPE'] == 'LakePond', 'total_line_length'], self.lmbda)


        # resistance
        df['resistance'] = self.resistance
        df.loc[df['farfield'], 'resistance'] = 0

        # resistance parameter (scenario)
        df['ScenResistance'] = self.ScenResistance
        df.loc[df['farfield'], 'ScenResistance'] = '__NONE__'

        # linesink location
        df.ix[df['FTYPE'] != 'LakePond', 'AutoSWIZC'] = 1 # Along stream centerline
        df.ix[df['FTYPE'] == 'LakePond', 'AutoSWIZC'] = 2 # Along surface water boundary


        # additional check to drop isolated lines
        isolated = [c for c in df.index if len(df.ix[c].dncomid) == 0 and len(df.ix[c].upcomids) == 0 and c not in self.wblist]
        df = df.drop(isolated, axis=0)

        '''
        print "removing any overlapping lines caused by simplication..."
        def actually_crosses(A, B, precis=0.0001):
            """A hybrid spatial predicate that determines if two geometries cross on both sides"""
            # from http://gis.stackexchange.com/questions/26443/is-there-a-way-to-tell-if-two-linestrings-really-intersect-in-jts-or-geos
            A = LineString(A) # convert back to shapely linestring for test below
            B = LineString(B)
            return (B.crosses(A) and
                    B.crosses(A.parallel_offset(precis, 'left')) and
                    B.crosses(A.parallel_offset(precis, 'right')))

        pdf = PdfPages('crossing_linesinks.pdf')
        crossing_lines = []
        fixed = []
        lines2drop = []
        for comid in df.index:
            if comid in fixed:
                continue
            crossed = df.ix[[actually_crosses(df.ix[comid, 'ls_coords'], l) for l in df.ls_coords]]
            # drop all overlapping lines but the largest
            crossed = crossed.append(df.ix[comid]).sort('ArbolateSu', ascending=False)
            if len(crossed) > 1:
                plt.figure()
                plt.hold(True)
                for cid in crossed.index:
                    plt.plot(LineString(df.ix[cid, 'ls_coords']).coords.xy[0], LineString(df.ix[cid, 'ls_coords']).coords.xy[1])
                plt.title(str(comid))


                # for cases where a lake is being overlapped by a tributary, move the tributary end vertex to outside the lake
                if 'LakePond' in list(crossed.FTYPE):
                    crossing_lines = crossed.ix[crossed.FTYPE != 'LakePond'].index
                    lake_comid = crossed.ix[crossed.FTYPE == 'LakePond'].index[0]
                    # for all tributaries to the lake (i.e. lines that don't represent lakes)
                    for line_comid in crossing_lines:
                        # get point where the line crosses the lake edge (convert from coordinates back to LineString for intersect)
                        intersection = LineString(df.ix[lake_comid].ls_coords).intersection(LineString(df.ix[line_comid].ls_coords)).xy
                        # move end of overlapping line to other side of intersection
                        next_to_last_vertex = df.ix[line_comid].ls_coords[-2]
                        diff = np.array(next_to_last_vertex) - np.ravel(intersection)
                        # pick a new endpoint that is between the intersection and next to last
                        new_endvert = tuple(np.array(next_to_last_vertex) - 0.9 * diff)
                        new_coords = df.ix[line_comid].ls_coords[:-1] + [new_endvert]
                        df.set_value(line_comid, 'ls_coords', new_coords)
                        plt.plot(LineString(df.ix[line_comid, 'ls_coords']).coords.xy[0], LineString(df.ix[line_comid, 'ls_coords']).coords.xy[1])

                # otherwise, if just lines are involved, drop the lines with the lowest Arbolate sums
                #else:
                    #df = df.drop(crossed.index[1:])

                # only address each comid once
                for comid in crossed.index:
                    fixed.append(comid)
                pdf.savefig()
                plt.close()
        pdf.close()
        '''
        # names
        df['ls_name'] = len(df)*[None]
        df['ls_name'] = df.apply(name, axis=1)


        # compare number of line segments before and after
        npoints_orig = sum([len(p)-1 for p in df['geometry'].map(lambda x: x.xy[0])])
        npoints_simp = sum([len(p)-1 for p in df.ls_coords])

        print '\nnumber of lines in original NHD linework: {}'.format(npoints_orig)
        print 'number of simplified lines: {}\n'.format(npoints_simp)


        if self.split_by_HUC:
            print '\nGrouping segments by hydrologic unit...'
            # intersect lines with HUCs; then group dataframe by HUCs
            HUCs_df = GISio.shp2df(self.HUC_shp, index=self.HUC_name_field, geometry=True)
            df[self.HUC_name_field] = len(df)*[None]
            for HUC in HUCs_df.index:
                lines = [line.intersects(HUCs_df.ix[HUC, 'geometry']) for line in df['geometry']]
                df.loc[lines, self.HUC_name_field] = HUC
            dfg = df.groupby(self.HUC_name_field)

            # write lines for each HUC to separate lss file
            HUCs = np.unique(df.HUC)
            for HUC in HUCs:
                dfh = dfg.get_group(HUC)
                outfile = '{}_{}.lss.xml'.format(self.outfile_basename, HUC)
                write_lss(dfh, outfile)
        else:
            write_lss(df, '{}.lss.xml'.format(self.outfile_basename))


        # write shapefile of results
        # convert lists in dn and upcomid columns to strings (for writing to shp)
        df['dncomid'] = df['dncomid'].map(lambda x: ' '.join([str(c) for c in x])) # handles empties
        df['upcomids'] = df['upcomids'].map(lambda x: ' '.join([str(c) for c in x]))

        # recreate shapely geometries from coordinates column; drop all other coords/geometries
        #df = df.drop([c for c in df.columns if 'geometry' in c], axis=1)
        df['geometry'] = df['ls_geom']
        df = df.drop(['ls_geom', 'ls_coords'], axis=1)
        #df['geometry'] = df['ls_coords'].map(lambda x: LineString(x))
        #df = df.drop(['ls_coords'], axis=1)
        GISio.df2shp(df, self.outfile_basename.split('.')[0]+'.shp', 'geometry', self.flowlines[:-4]+'.prj')

        self.efp.close()
        print 'Done!'


class InputFileMissing(Exception):
    def __init__(self, infile):
        self.infile = infile
    def __str__(self):
        return('\n\nCould not open or parse input file {0}.\nCheck for errors in XML formatting.'.format(self.infile))