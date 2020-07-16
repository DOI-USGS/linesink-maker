import lsmaker

input_file = 'Medford_lines.xml'
ls = lsmaker.LinesinkData(input_file)

ls.preprocess(save=True)

ls.make_linesinks(shp='preprocessed/lines.shp')
