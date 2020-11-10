def read_lines(file_path):
    with open(file_path) as f:
        lines = [line.rstrip() for line in f]
    return lines
