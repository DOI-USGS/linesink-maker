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
    
    df[['x1', 'x2']] = df[['x1', 'x2']] * coords_mult + solver_x0
    df[['y1', 'y2']] = df[['y1', 'y2']] * coords_mult + solver_y0
    df['geometry'] = [LineString([(r.x1, r.y1), (r.x2, r.y2)]) for i, r in df.iterrows()]
    df2shp(df, outshp, epsg=epsg)