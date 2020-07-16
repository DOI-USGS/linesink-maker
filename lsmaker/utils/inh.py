"""
Utility for working with GFLOW inhomogeneities
"""
import pandas as pd
try:
    import shapefile
except:
    pass
from shapely.geometry import shape


class Inhomogeneity:
    fields = ['Label',
              'HydCond',
              'BottomElevation',
              'AverageHead',
              'Recharge',
              'Porosity',
              'Color',
              'ChangeK',
              'ChangeR',
              'ChangeP',
              'ChangeB',
              'ChangeA',
              'ScenHydCond',
              'ScenRecharge',
              'chkScenarioHydCond',
              'chkScenarioRecharge',
              'DefaultHydCond',
              'DefaultRecharge']
    tab = ' ' * 4

    def __init__(self, geometry=None,
                 Label='None',
                 HydCond=1.,
                 BottomElevation=0.,
                 AverageHead=10000,
                 Recharge=0.,
                 Porosity=0.2,
                 Color=65535,
                 ChangeK=True,
                 ChangeR=False,
                 ChangeP=False,
                 ChangeB=False,
                 ChangeA=False,
                 ScenHydCond='__NONE__',
                 ScenRecharge='__NONE__',
                 chkScenarioHydCond=True,
                 chkScenarioRecharge=False,
                 DefaultHydCond=0,
                 DefaultRecharge=0):
        self.Label = Label
        self.HydCond = HydCond
        self.BottomElevation = BottomElevation
        self.AverageHead = AverageHead
        self.Recharge = Recharge
        self.Porosity = Porosity
        self.Color = Color
        self.ChangeK = ChangeK
        self.ChangeR = ChangeR
        self.ChangeP = ChangeP
        self.ChangeB = ChangeB
        self.ChangeA = ChangeA
        self.ScenHydCond = ScenHydCond
        self.ScenRecharge = ScenRecharge
        self.chkScenarioHydCond = chkScenarioHydCond
        self.chkScenarioRecharge = chkScenarioRecharge
        self.DefaultHydCond = DefaultHydCond
        self.DefaultRecharge = DefaultRecharge

        if isinstance(geometry, list):
            self.Vertices = geometry
        elif isinstance(geometry, Polygon):
            self.Vertices = list(geometry.exterior.coords)
        elif isinstance(geometry, LineString):
            self.Vertices = list(geometry.coords)
        else:
            print('unknown geometry type')

    def to_xml(self, indent=1):
        #Need to check for duplicate vertices (i.e. at start/end!)
        def tab(ntab=0):
            return self.tab * indent + self.tab * ntab
        text = '\n{}<Inhomogeneity>\n'.format(tab())
        for field in self.fields:
            v = str(self.__dict__[field]).lower() # convert t/f
            text += tab(1) + '<{0}>{1}</{0}>\n'.format(field, v)
        text += tab(1) + '<Vertices>\n'
        for x, y in self.Vertices:
            text += tab(2) + '<Vertex>\n'
            text += tab(3) + '<X>{:.2f}</X>\n'.format(x)
            text += tab(3) + '<Y>{:.2f}</Y>\n'.format(y)
            text += tab(2) + '</Vertex>\n'
        text += tab(1) + '</Vertices>\n'
        text += tab() + '</Inhomogeneity>\n'
        return text

def readshp(inhshp):
    sf = shapefile.Reader(inhshp)
    geoms = [shape(s) for s in sf.iterShapes()]
    attributes = list(sf.iterRecords())
    field_names = [c[0] for c in sf.fields[1:]]
    df = pd.DataFrame(attributes, columns=field_names)
    df['geometry'] = geoms
    return df