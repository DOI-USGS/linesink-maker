from distutils.core import setup

DESCRIPTION = """\
Rapidly construct linesink networks for the analytic element groundwater flow modeling
program GFLOW. Reads information from NHDPlus v2, which is specified in an XML input 
file. Writes a GFLOW linesink string (lss) XML import file, and a shapefile of the 
linesink network.
"""

def run():
    setup(name="lsmaker",
          version="0.1",
          description="Rapidly construct GFLOW models from NHDPlus v2.",
          author="Andy Leaf",
          packages=[""],
          long_descripton=DESCRIPTION,
          )
          
if __name__ == "__main__":
    run()