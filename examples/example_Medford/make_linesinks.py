import lsmaker

input_file = 'Medford_lines.xml'
ls = lsmaker.linesinks(input_file)

ls.preprocess(save=True)

ls.makeLineSinks(shp='preprocessed/lines.shp')
