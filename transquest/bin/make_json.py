import argparse
import json
import pandas as pd

import random


def main(inpf, outf, mix=False):
    data = pd.read_csv(inpf, sep='\t', quoting=3)
    src = data['original']
    tgt = data['translation']
    if mix:
        random.shuffle(tgt)
    data_list = []
    for srci, tgti in zip(src, tgt):
        data_list.append(
            {
                'text_a': srci,
                'text_b': tgti,
            }
        )
    json.dump({'data': data_list}, open(outf, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--mix', default=False, action='store_true')
    args = parser.parse_args()
    main(args.input, args.output, mix=args.mix)
