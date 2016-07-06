import sys
sys.path.append('../lsmaker')
import os
import shutil
import numpy as np
import pandas as pd
from shapely.geometry import LineString

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


def plot_flooding(grdfile, dem, epsg,
                  outpath='',
                  clipto=None,
                  solver_x0=0, solver_y0=0, scale_xy=0.3048,
                  dem_mult=1):

    from GISops import project_raster, clip_raster, _to_geojson
    try:
        import rasterio
        from rasterio import transform
        from rasterio.warp import reproject, Resampling
        from rasterio.tools.mask import mask
    except ImportError:
        print('This method requires Rasterio')
        return

    # make a temporary folder to house all the cruft
    tmpath = outpath + '/tmp/'
    if not os.path.isdir(tmpath):
        os.makedirs(tmpath)

    # convert the heads surfer grid to raster
    solver_x0 = solver_x0
    solver_y0 = solver_y0
    wtfile = os.path.join(outpath, 'heads_prj.tiff')
    print('reading {}...'.format(grdfile))
    hds = surferGrid(grdfile)
    hds.scale_xy(scale_xy)
    hds.offset_xy(solver_x0, solver_y0)
    hds.write_raster(wtfile, epsg=epsg)

    clipto = _to_geojson(clipto) # convert input to geojson

    out = os.path.join(tmpath, 'heads_rs.tif')
    demcp = os.path.join(tmpath, 'dem_cp.tif')
    out2 = os.path.join(tmpath, 'heads_cp.tif')

    with rasterio.open(dem) as demobj:
        project_raster(wtfile, out, dst_crs='epsg:{}'.format(epsg), resampling=1, resolution=demobj.res)
        clip_raster(dem, clipto, demcp)
        clip_raster(out, clipto, out2)

    with rasterio.open(demcp) as demcpobj:
        with rasterio.open(out2) as hds:
            dtw = demcpobj.read(1) - hds.read(1)
            dtw[dtw < -1e4] = np.nan
            fld = dtw.copy()
            fld[fld > 0] = np.nan
            out_meta = demcpobj.meta.copy()
            out_meta.update({'dtype': 'float64', 'compress': 'LZW'})

            out_dtw = outpath+'/dtw.tif'
            out_fld = outpath+'/flooding.tif'
            with rasterio.open(out_dtw, "w", **out_meta) as dest:
                dest.write(dtw, 1)
                print('wrote {}'.format(out_dtw))
            with rasterio.open(out_fld, "w", **out_meta) as dest:
                dest.write(fld, 1)
                print('wrote {}'.format(out_fld))
    # garbage cleanup
    shutil.rmtree(tmpath)

def plot_flooding_arcpy(grdfile, dem,
                        outpath='flooding',
                        solver_x0=0, solver_y0=0, scale_xy=0.3048, epsg=None,
                        dem_mult=0.3048, nearfield=None,
                        resample_heads=True, resample_cellsize=10):
    """Subtract gridded heads from GFLOW from DEM; save results as raster.
    (requires arcpy and rasterio)

    gridfile: str
        Gridded heads file output from GFLOW (*.GRD)
    dem:
        DEM for model area.
    outpath :
        Folder for writing output raster(s) to.
    solver_x0 : float
    solver_y0 : float
        To get the solve origin (solver_x0, solver_y0), in GFLOW choose Tools > GFLOW Database Viewer,
        then View > Base Tables > Model.
    scale_xy : float
        Multiplier to convert GFLOW coordinates to GIS coordinates.
    epsg : int
        EPSG code for GIS projected coordinate system (e.g. 26715 for NAD27 UTM zone 15)
    dem_mult : float
        Multiplier from DEM elevation units to GFLOW units.
    nearfield : str
        Shapefile of model nearfield.
    resample_heads : boolean
        Resample gridded GFLOW heads (typically coarse) to finer resolution using bilinear interpolation).
    resample_cellsize : float
        Cellsize for resampled heads (typically same as DEM resolution).

    Notes
    =====

    """
    try:
        import arcpy
    except:
        print('Method requires arcpy.')
        return

    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    arcpy.env.workspace = outpath
    #arcpy.Delete_management('*')
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension('Spatial')

    print('reading {}...'.format(dem))
    demft = arcpy.Raster(dem) * dem_mult
    solver_x0 = solver_x0
    solver_y0 = solver_y0
    wtfile = os.path.join(outpath,'heads_prj.tiff')
    print('reading {}...'.format(grdfile))
    hds = surferGrid(grdfile)
    hds.scale_xy(scale_xy)
    hds.offset_xy(solver_x0, solver_y0)
    hds.write_raster(wtfile, epsg=epsg)

    arcpy.Resample_management(wtfile, 'wtfile_rs', resample_cellsize, "BILINEAR")
    flooding = arcpy.sa.Minus(demft, 'wtfile_rs')
    arcpy.MakeFeatureLayer_management(nearfield, 'nf')
    arcpy.Clip_management(flooding,'#', 'fldcp', nearfield, "#", "ClippingGeometry")
    flooding = arcpy.sa.SetNull('fldcp', 'fldcp', "VALUE > 0")
    print('writing {}/flooding...'.format(outpath))
    flooding.save('flooding')

def write_streamflow_shapefile(xtr, outshp=None, solver_x0=0, solver_y0=0,
                               coords_mult=0.3048, epsg=None):
    """Read linesink results from GFLOW extract (.xtr) file and write to shapefile.
    To get the solve origin (solver_x0, solver_y0), in GFLOW choose Tools > GFLOW Database Viewer, 
    then View > Base Tables > Model.
    """
    from GISio import df2shp
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
            self.header = next(input)
            self.nrow, self.ncol = map(int, next(input).strip().split())
            self.xmin, self.xmax = map(float, next(input).strip().split())
            self.ymin, self.ymax = map(float, next(input).strip().split())
            self.zmin, self.zmax = map(float, next(input).strip().split())

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
        data = np.loadtxt(grdfile, skiprows=5)
        self.data = np.reshape(data, (self.ncol, self.nrow))

    def write_grd(self, fname='output.grd'):

        with open(fname, 'w') as output:
            header = self.header.strip() + '\n'
            header += '{:.0f} {:.0f}\n'.format(self.nrow, self.ncol)
            header += '{:.2f} {:.2f}\n'.format(self.xmin, self.xmax)
            header += '{:.2f} {:.2f}\n'.format(self.ymin, self.ymax)
            header += '{:.2f} {:.2f}'.format(self.zmin, self.zmax)
            np.savetxt(output, self.data, fmt='%.2f', delimiter=' ', header=header, comments='')

    def write_raster(self, fname='output', epsg=None):
        try:
            import rasterio
            from rasterio import transform
        except ImportError:
            print('This method requires Rasterio')
            return

        tfm = transform.from_bounds(self.xmin, self.ymin, self.xmax, self.ymax, self.ncol, self.nrow)
        if epsg is not None:
            crs = {'init': 'epsg:{}'.format(epsg)}
        else:
            crs = None

        with rasterio.drivers():
            with rasterio.open(fname,
                               'w',
                               driver='GTiff',
                               width=self.ncol,
                               height=self.nrow,
                               count=1,
                               dtype=np.float64,
                               nodata=0,
                               transform=tfm,
                               crs=crs) as dst:
                dst.write_band(1, np.flipud(self.data))