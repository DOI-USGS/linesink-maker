import os
from shapely.geometry import LineString
import gisutils


class Diagnostics:

    def __init__(self, lsm_object=None, ls_shapefile=None,
                 logfile='lsmaker-log.txt'):

        if lsm_object is not None:
            self.__dict__ = lsm_object.__dict__.copy()
        elif ls_shapefile is not None:
            self.df = gisutils.shp2df(ls_shapefile, index='COMID')
            self.prj = ls_shapefile[:-4] + '.prj'
        else:
            print('Provide either LinesinkMaker object of shapefile as input.')
            return

        self.diagnostics_file = logfile

    def check4crossing_lines(self):
        with open(self.diagnostics_file, 'a') as ofp:
            print('\nChecking for lines that cross...')
            ofp.write('\nChecking for lines that cross...')
            comids = self.df.COMID.values
            geoms = [LineString(ls_coords) for ls_coords in self.df.ls_coords]

            crosses = set()
            for comid, linesink in zip(comids, geoms):
                for other_comid, other_linesink in zip(comids, geoms):
                    if linesink.crosses(other_linesink):
                        crosses.add((comid, other_comid))

            if len(crosses) > 0:
                print('Warning! Crossing LinesinkData found. Check these before running GFLOW.\n' \
                    'See {} for more details'.format(self.diagnostics_file))
                ofp.write('\nThe following line segments cross, and should be fixed manually before running GFLOW:\n')
                for c in crosses:
                    ofp.write('{}, {}\n'.format(*c))
                ofp.write('\r\n')
            else:
                print('passed.\n')
        return crosses

    def check_vertices(self):
        with open(self.diagnostics_file, 'a') as ofp:
            print('\nChecking for duplicate vertices...')
            ofp.write('\nChecking for duplicate vertices...')
            if 'ls_coords' not in self.df.columns:
                # convert geometries to coordinates
                def xy_coords(x):
                    xy = list(zip(x.xy[0], x.xy[1]))
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
                print('Duplicate coordinates found at:\n{}'.format(duplicate_coords))
                print('See {}'.format(self.diagnostics_file))
                ofp.write('Duplicate coordinates:')
                for crd in duplicate_coords:
                    ofp.write('{} {}\n'.format(crd, self.df.ix[crd, 'ls_coords']))
            else:
                print('passed.\n')
                ofp.write('passed.\n')

    def check4zero_gradient(self, log=True):
        txt = ''
        msg = '\nChecking for lines with zero gradient...'
        print(msg)
        txt += msg
        self.df['dStage'] = self.df.maxElev - self.df.minElev
        comids0 = list(self.df[self.df['dStage'] == 0].index)
        if len(comids0) > 0:
            msg = '{} lines with zero gradient found\n'.format(len(comids0))
            print(msg)
            txt += msg
        else:
            msg = 'No zero-gradient lines found.\n'
            print(msg)
            txt += msg
        with open(self.diagnostics_file, 'a') as ofp:
            ofp.write(txt)
        return comids0