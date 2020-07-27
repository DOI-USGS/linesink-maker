import lsmaker

input_file = 'Medford_lines.yml'
ls = lsmaker.LinesinkData(input_file)
ls.make_linesinks()
