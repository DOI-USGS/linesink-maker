__author__ = 'aleaf'

import os
import numpy as np
from shapely.geometry import LineString
import GISio

class Diagnostics:

    def __init__(self, lsm_object=None, ls_shapefile=None):

        if lsm_object is not None:
            self.__dict__ = lsm_object.__dict__.copy()
        elif ls_shapefile is not None:
            self.df = GISio.shp2df(ls_shapefile, index='COMID')
            self.prj = ls_shapefile[:-4] + '.prj'
        else:
            print 'Provide either LinesinkMaker object of shapefile as input.'
            return

        self.diagnostics_file = 'lsm_diagnostics.txt'
        if os.path.exists(self.diagnostics_file):
            os.remove(self.diagnostics_file)

    def check4crossing_lines(self):
        ofp = open(self.diagnostics_file, 'a')
        print 'Checking for lines that cross...'
        comids = self.df.index.values
        geoms = self.df.geometry.tolist()

        crosses = set()
        for i, linesink in enumerate(geoms):
            [crosses.add(comids[j]) for j, g in enumerate(geoms) if linesink.crosses(g)]

        if len(crosses) > 0:
            print 'Warning! Crossing linesinks found. These will have to be modified before running GFLOW.\n' \
                  'See {}'.format(self.diagnostics_file)
            ofp.write('The following line segments cross, and should be fixed manually before running GFLOW:\n')
            for c in crosses:
                ofp.write('{} '.format(c))
            ofp.write('\r\n')
        else:
            print 'passed.'
        ofp.close()
        return crosses

    def check_vertices(self):
        print 'Checking for duplicate vertices...'
        if 'ls_coords' not in self.df.columns:
            # convert geometries to coordinates
            def xy_coords(x):
                xy = zip(x.xy[0], x.xy[1])
                return xy

            # add column of lists, containing linesink coordinates
            self.df.loc[:, 'ls_coords'] = self.df.geometry.apply(xy_coords)
        '''
        duplicate_coords = [self.df.index.values[i]
                            for i, crds in enumerate(self.df.ls_coords.tolist())
                            if len(set(crds)) != len(crds)]
        '''
        all_coords = []
        [[all_coords.append(c) for c in l] for l in self.df.ls_coords.tolist()]
        duplicate_coords = [self.df.index.values[i]
                            for i, crds in enumerate(self.df.ls_coords.tolist())
                            if all_coords.count(crds) > 1]
        if len(duplicate_coords) > 0:
            print 'Duplicate coordinates found at:\n{}'.format(duplicate_coords)
            print 'See {}'.format(self.diagnostics_file)
            ofp = open(self.diagnostics_file, 'a')
            ofp.write('Duplicate coordinates:')
            for crd in duplicate_coords:
                ofp.write('{} {}\n'.format(crd, self.df.ix[crd, 'ls_coords']))
            ofp.close()
        else:
            print 'passed.'

    def check4zero_gradient(self):

        print 'Checking for lines with zero gradient...'
        self.df['dStage'] = self.df.maxElev - self.df.minElev
        comids0 = list(self.df[self.df['dStage'] == 0].index)
        if len(comids0) > 0:
            print '{} lines with zero gradient found'.format(len(comids0))
        else:
            print 'No zero-gradient lines found.'
        return comids0