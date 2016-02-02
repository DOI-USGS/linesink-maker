import sys
sys.path.append('../lsmaker')
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from GISio import df2shp

def get_ls_skiprows_nrows(xtr):
    with open(xtr) as f:
        knt = 0
        while True:
            line = next(f)
            knt += 1
            if '! head specified line sinks' in line:
                skiprows = knt + 1
            if '! stream linkages' in line:
                nrows = knt - skiprows -1
                break
    return skiprows, nrows

def get_linesink_results(xtr):
    """Read linesink results from GFLOW extract (.xtr) file into dataframe.
    """
    names = ['x1',
             'y1',
             'x2',
             'y2',
             'spec_head',
             'calc_head',
             'discharge',
             'width',
             'resistance',
             'depth',
             'baseflow',
             'overlandflow',
             'BC_pct_err',
             'label'] 
    ls_skiprows, nrows = get_ls_skiprows_nrows(xtr)
    
    return pd.read_csv(xtr, skiprows=ls_skiprows, nrows=nrows, header=None, names=names)

def write_streamflow_shapefile(xtr, outshp=None, solver_x0=0, solver_y0=0,
                               coords_mult=0.3048, epsg=None):
    """Read linesink results from GFLOW extract (.xtr) file and write to shapefile.
    To get the solve origin (solver_x0, solver_y0), in GFLOW choose Tools > GFLOW Database Viewer, 
    then View > Base Tables > Model.
    """
    if outshp is None:
        outshp = '{}_streamflow.shp'.format(xtr[:-4])
    df = get_linesink_results(xtr)
    df = df[df.resistance.values > 0].copy() # only include
    df[['x1', 'x2']] = df[['x1', 'x2']] * coords_mult + solver_x0
    df[['y1', 'y2']] = df[['y1', 'y2']] * coords_mult + solver_y0
    df['geometry'] = [LineString([(r.x1, r.y1), (r.x2, r.y2)]) for i, r in df.iterrows()]
    df2shp(df, outshp, epsg=epsg)

class surferGrid:
    def __init__(self, grdfile=None, data=None):
        self.header = None
        self.nrow = None
        self.ncol = None
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.zmax = None
        self.zmin = None
        self.data = data
        if grdfile is not None:
            self.read_grd(grdfile)

    def _read_grd_header(self, grdfile):
        with open(grdfile) as input:
            self.header = input.next()
            self.nrow, self.ncol = map(int, input.next().strip().split())
            self.xmin, self.xmax = map(float, input.next().strip().split())
            self.ymin, self.ymax = map(float, input.next().strip().split())
            self.zmin, self.zmax = map(float, input.next().strip().split())

    def scale_xy(self, mult=1):
        self.xmin *= mult
        self.xmax *= mult
        self.ymin *= mult
        self.ymax *= mult

    def scale_z(self, mult=1):
        self.zmin *= mult
        self.zmax *= mult

    def offset_xy(self, x0=0, y0=0):
        self.xmin += x0
        self.xmax += x0
        self.ymin += y0
        self.ymax += y0

    def read_grd(self, grdfile):
        self._read_grd_header(grdfile)
        data = np.loadtxt(grd, skiprows=5)
        self.data = np.reshape(data, (self.ncol, self.nrow))

    def write_grd(self, fname='output.grd'):

        with open(fname, 'w') as output:
            header = self.header.strip() + '\n'
            header += '{:.0f} {:.0f}\n'.format(self.nrow, self.ncol)
            header += '{:.2f} {:.2f}\n'.format(self.xmin, self.xmax)
            header += '{:.2f} {:.2f}\n'.format(self.ymin, self.ymax)
            header += '{:.2f} {:.2f}'.format(self.zmin, self.zmax)
            np.savetxt(output, self.data, fmt='%.2f', delimiter=' ', header=header, comments='')