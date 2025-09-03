import pandas

def from_file(filename, delimiter):
    df = pandas.read_csv(filename, delimiter = delimiter)
    return df

