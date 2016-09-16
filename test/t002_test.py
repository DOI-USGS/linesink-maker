"""
Test utils
"""
import os
from shapely.ops import unary_union
import pandas as pd
import lsmaker
from GISio import shp2df, df2shp

def test_plot_flooding_PF():
    """Tests the plot_flooding utility.
    (requires Park Falls grid and dem to run)"""
    solver_x0 = 227540.11 # origin of GFLOW solver coordinates in NAD 27 UTM 16
    solver_y0 = 5042425.36
    epsg = 26716
    path = 'D:/ATLData/USFS/ParkFalls/'
    grd = path + 'PF15.GRD'
    dem = path + 'dem/dem_utm_ft'
    output_folder = path + 'PF15_results/'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    output_flooded_heads_file = output_folder + 'flooding'
    lines = shp2df(path + 'shps/PF14_lines.shp')
    alllines = unary_union(lines.geometry)
    farfield = alllines.convex_hull
    lsmaker.utils.plot_flooding(grd, dem=dem, epsg=epsg, clipto=[farfield],
                                      outpath=output_flooded_heads_file,
                                      solver_x0=solver_x0, solver_y0=solver_y0, scale_xy=0.3048,
                                      dem_mult=1)

def test_plot_flooding_NS():
    """Tests the plot_flooding utility.
    (requires Nicolet grid and dem to run)"""
    solver_x0 = 268548.41 # origin of GFLOW solver coordinates in NAD 27 UTM 16
    solver_y0 = 4934415.64
    epsg = 26716
    path = 'D:/ATLData/USFS/Nicolet/Nicolet_south/'
    grd = path + 'NS1.GRD'
    dem = path + '../dem/dem_utm_ft'
    aquifer_bottom = path + '../from_WGNHS/Bedrock/TopoToR_Nico12.tif'
    output_folder = path + '../NS1_results/'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    output_flooded_heads_file = output_folder + 'flooding'
    farfield = path + '../shps/Nicolet_south_ff.shp'
    lsmaker.utils.plot_flooding(grd, dem=dem, epsg=epsg, clipto=farfield,
                                aquifer_bottom=aquifer_bottom,
                                      outpath=output_flooded_heads_file,
                                      solver_x0=solver_x0, solver_y0=solver_y0, scale_xy=0.3048,
                                      dem_mult=1)
def test_plot_flooding():
    """Tests the plot_flooding utility."""
    solver_x0 = 671467.1 # origin of GFLOW solver coordinates in NAD 27 UTM 16
    solver_y0 = 4997427.91
    epsg = 26715
    path = 'input/'
    grd = 'output/test.GRD'
    dem = path + 'dem.tif'
    output_folder = 'output/'
    clipto = path + 'testnearfield.shp'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    output_flooded_heads_file = output_folder + 'flooding'
    lsmaker.utils.plot_flooding(grd, dem=dem, epsg=epsg, clipto=clipto,
                                      outpath=output_flooded_heads_file,
                                      solver_x0=solver_x0, solver_y0=solver_y0, scale_xy=0.3048,
                                      dem_mult=1/.3048)

if __name__ == '__main__':
    #test_plot_flooding_PF()
    test_plot_flooding_NS()
    #test_plot_flooding()