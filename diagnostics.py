__author__ = 'aleaf'

from shapely.geometry import LineString
import GISio


def actually_crosses(A, B, precis=0.0001):
    """A hybrid spatial predicate that determines if two geometries cross on both sides"""
    # from http://gis.stackexchange.com/questions/26443/is-there-a-way-to-tell-if-two-linestrings-really-intersect-in-jts-or-geos
    return (B.crosses(A) and
            B.crosses(A.parallel_offset(precis, 'left')) and
            B.crosses(A.parallel_offset(precis, 'right')))

class Diagnostics:

    def __init__(self, lsm_object=None, ls_shapefile=None):

        if lsm_object is not None:
            self.__dict__ = self.lsm_object.__dict__.copy()
        elif ls_shapefile is not None:
            self.df = GISio.shp2df(ls_shapefile, index='COMID')
            self.prj = ls_shapefile[:-4] + '.prj'
        else:
            print 'Provide either LinesinkMaker object of shapefile as input.'
            return

    def check4crossing_lines(self):

        comids = self.df.index.values
        geoms = self.df.geometry.tolist()

        for linesink in geoms:
            crosses = [comids[i] if linesink.intersects(g) for i, g in enumerate(geoms)]
