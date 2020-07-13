import json
import pandas as pd

import sys


def main(inpf, outf):
    data = pd.read_csv(inpf, sep='\t', quoting=3)
    data_list = []
    for i, row in data.iterrows():
        data_list.append(
            {
                'text_a': row['original'],
                'text_b': row['translation'],
            }
        )
    json.dump({'data': data_list}, open(outf, 'w'))


if __name__ == '__main__':
    inp_file = sys.argv[1]
    out_file = sys.argv[2]
    main(inp_file, out_file)
