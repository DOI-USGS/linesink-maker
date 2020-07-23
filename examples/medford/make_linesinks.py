import lsmaker

input_file = 'Medford_lines.yml'
ls = lsmaker.LinesinkData(input_file)

ls.preprocess(save=True)

ls.make_linesinks(shp='preprocessed/lines.shp')
