import argparse
import os
import sys

import torch

from transquest.algo.transformers.run_model import QuestModel

from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetWordLevel
from transquest.data.mapping_tokens_bpe import map_pieces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--src_file', required=True)
    parser.add_argument('--tgt_file', required=True)
    parser.add_argument('--tags_file', required=True)
    parser.add_argument('--out_file', required=False, default=None)
    parser.add_argument('--output_dir', required=False, default=None)
    args = parser.parse_args()
    config = load_config(args)
    config['model_name'] = args.model_dir
    test_set = DatasetWordLevel(config, evaluate=True)
    test_data = test_set.make_dataset(
        os.path.join(args.src_file),
        os.path.join(args.tgt_file),
        os.path.join(args.tags_file),
        no_cache=True,
    )
    assert os.path.isdir(args.model_dir)
    model = QuestModel(config['model_type'], args.model_dir, use_cuda=torch.cuda.is_available(), args=config)
    _, preds = model.evaluate(test_data)
    res = []
    for i, preds_i in enumerate(preds):
        bpe_pieces = test_set.tokenizer.tokenize(test_set.examples[i].tgt)
        mt_tokens = test_set.examples[i].tgt
        mapped = map_pieces(bpe_pieces, mt_tokens, preds_i, 'average')
        res.append([int(v) for v in mapped])
    outf = open(args.out_file, 'w') if args.out_file is not None else sys.stdout
    for pred in preds:
        outf.write('{}\n'.format(' '.join([str(p) for p in pred])))


if __name__ == '__main__':
    main()
