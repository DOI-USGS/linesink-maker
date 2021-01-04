import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from shapely.geometry import LineString
import gisutils


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


def write_heads_raster(grdfile, outraster='heads.tiff',
                       solver_x0=0, solver_y0=0, scale_xy=.3048,
                       epsg=26715):
    hds = SurferGrid(grdfile)
    hds.scale_xy(scale_xy)
    hds.offset_xy(solver_x0, solver_y0)
    hds.write_raster(outraster, epsg=epsg)


def plot_flooding(grdfile, dem, epsg,
                  aquifer_bottom=None,
                  outpath='',
                  clipto=None,
                  solver_x0=0, solver_y0=0, scale_xy=0.3048,
                  dem_mult=1, resolution=(30, 30)):
    """Compare heads simulated by GFLOW to land surface.

    Parameters
    ----------
    gridfile : str
        Surfer .GRD file output by GFLOW GUI after a solution.
        Note that the raw .GRD file output by the GUI will be in
        model units, hence scale_xy would be 0.3048 if the model
        is in feet and GIS in meters. If the surfer grid
        was exported by the GFLOW GUI (under Tools > Export),
        the GFLOW has already done any conversion, so scale_xy should be 1.
    dem : str
        DEM raster file.
    epsg : str
        EPSG code. Must be consistent with coordinate system of clipto feature.
    aquifer_bottom : str
        Raster file of aquifer bottom elevations. Used to compute saturated thickness.
    outpath : str
        folder for saving the output rasters
    clipto : str, or list
        Feature(s) defining the extent of the output flooding and depth to water rasters.
        Can be a shapefile or list of shapely or geojson objects.
    solver_x0 : float
    solver_y0 : float
        To get the solve origin (solver_x0, solver_y0), in GFLOW choose Tools > GFLOW Database Viewer,
        then View > Base Tables > Model.
    scale_xy : float
        Multiplier to convert GFLOW coordinates to GIS coordinates. See notes above regarding .GRD file.
    dem_mult : float
        Multiplier from DEM elevation units to GFLOW units.
    resolution : tuple of length 2
        (x, y) resolution of output rasters. Must be chosen carefully with coordinate system.
        Default is (30, 30).
    """
    try:
        import rasterio
        from rasterio import transform
        from rasterio.warp import reproject, Resampling
        from rasterio.mask import mask
        from rasterio.crs import CRS
        from gisutils.raster import clip_raster, project_raster
    except ImportError as e:
        print('This method requires rasterio and the raster module of gis-utils.')
        raise Exception(e)

    # make a temporary folder to house all the cruft
    outpath = Path(outpath)
    tmpath =outpath / 'tmp'
    tmpath.mkdir(parents=True, exist_ok=True)

    # convert the heads surfer grid to raster
    solver_x0 = solver_x0
    solver_y0 = solver_y0
    wtfile = os.path.join(outpath, 'heads_prj.tif')
    write_heads_raster(grdfile, wtfile,
                       solver_x0=solver_x0, solver_y0=solver_y0,
                       scale_xy=scale_xy, epsg=epsg)

    # clipto must be a list (should add conversion if not)
    #clipto = _to_geojson(clipto) # convert input to geojson

    heads_rs = tmpath / 'heads_rs.tif'
    dem_rs = tmpath / 'dem_rs.tif'
    dem_cp = tmpath / 'dem_cp.tif'
    heads_cp = outpath / 'heads_cp.tif'#os.path.join(tmpath, 'heads_cp.tif')
    aq_bot_rs = tmpath / 'botm_rs.tif'
    aq_bot_cp = tmpath / 'botm_cp.tif'

    project_raster(wtfile, heads_rs, dest_crs='epsg:{}'.format(epsg), resampling=1, resolution=resolution)
    project_raster(dem, dem_rs, dest_crs='epsg:{}'.format(epsg), resampling=1, resolution=resolution)
    clip_raster(dem_rs, clipto, dem_cp)
    clip_raster(heads_rs, clipto, heads_cp)
    if aquifer_bottom is not None:
        project_raster(aquifer_bottom, aq_bot_rs, dest_crs='epsg:{}'.format(epsg),
                       resampling=1, resolution=resolution)
        clip_raster(aq_bot_rs, clipto, aq_bot_cp)

    with rasterio.open(dem_cp) as demcpobj:
        with rasterio.open(heads_cp) as hds:
            # rasters might be of slightly different shape after clipping
            # (depending on original offset(s)?)
            # slice both to minimum dimmensions
            demarr, hdsarr = demcpobj.read(1), hds.read(1)
            h = np.min((demcpobj.height, hds.height))
            w = np.min((demcpobj.width, hds.width))
            dtw = demarr[:h, :w] * dem_mult - hdsarr[:h, :w]
            dtw[dtw < -1e4] = np.nan
            fld = dtw.copy()
            fld[fld > 0] = np.nan
            out_meta = demcpobj.meta.copy()
            out_meta.update({'dtype': 'float64', 'compress': 'LZW',
                             'width': w, 'height': h})

            out_dtw = outpath / 'dtw.tif'
            out_fld = outpath / 'flooding.tif'
            out_sth = outpath / 'sat_thickness.tif'
            with rasterio.open(out_dtw, "w", **out_meta) as dest:
                dest.write(dtw, 1)
                print('wrote {}'.format(out_dtw))
            with rasterio.open(out_fld, "w", **out_meta) as dest:
                dest.write(fld, 1)
                print('wrote {}'.format(out_fld))
            # compute and write sat. thickness
            if aquifer_bottom is not None:
                with rasterio.open(aq_bot_cp) as aqbot:
                    h = np.min((h, aqbot.height))
                    w = np.min((w, aqbot.width))
                    botm = aqbot.read(1)
                    botm = botm[:h, :w]
                    # force aquifer bottom to be no higher than land surface
                    botm[botm > demarr[:h, :w]] = demarr[botm > demarr[:h, :w]]
                    sat_thickness = hdsarr[:h, :w] - botm[:h, :w]
                    # where there is flooding, limit sat. thickness to surficial deposit thickness
                    sat_thickness[dtw[:h, :w] < 0] = demarr[dtw[:h, :w] < 0] - botm[dtw[:h, :w] < 0]
                    sat_thickness[sat_thickness > 1e4] = np.nan # no data values
                    # set negative values resulting from flooding where botm = land surface to 0
                    sat_thickness[sat_thickness < 0] = 0
                    out_meta.update({'width': w, 'height': h})
                    with rasterio.open(out_sth, "w", **out_meta) as dest:
                        dest.write(sat_thickness, 1)
                        print('wrote {}'.format(out_sth))

    # garbage cleanup
    shutil.rmtree(tmpath)


def write_streamflow_shapefile(xtr, outshp=None, solver_x0=0, solver_y0=0,
                               coords_mult=0.3048, crs=None, **kwargs):
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
    gisutils.df2shp(df, outshp, crs=crs, **kwargs)


class SurferGrid:
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
        print('reading {}...'.format(grdfile))
        with open(grdfile) as input:
            self.header = next(input)
            self.ncol, self.nrow = map(int, next(input).strip().split())
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
        try:
            data = np.loadtxt(grdfile, skiprows=5)
        except ValueError: # handle uneven number of columns
            with open(grdfile) as input:
                all_data = [l.split() for l in input.readlines()][5:]
                data = np.hstack(all_data).astype(float)

        self.data = np.reshape(data, (self.nrow, self.ncol))

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

        with rasterio.Env():
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
        print('wrote {}.'.format(fname))