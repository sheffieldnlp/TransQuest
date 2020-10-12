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
    test_set.make_dataset(
        os.path.join(args.src_file),
        os.path.join(args.tgt_file),
        os.path.join(args.tags_file),
        no_cache=True,
    )
    assert os.path.isdir(args.model_dir)
    model = QuestModel(config['model_type'], args.model_dir, use_cuda=torch.cuda.is_available(), args=config)
    _, preds = model.evaluate(test_set.tensors)
    res = []
    for i, preds_i in enumerate(preds):
        input_ids = test_set.tensors.tensors[0][i]
        input_mask = test_set.tensors.tensors[1][i]
        preds_i = [p for j, p in enumerate(preds_i) if input_mask[j] and input_ids[j] not in (0, 2)]
        bpe_pieces = test_set.tokenizer.tokenize(test_set.examples[i].text_a)
        mt_tokens = test_set.examples[i].text_a
        mapped = map_pieces(bpe_pieces, mt_tokens, preds_i, 'average', from_sep='▁')
        res.append([int(v) for v in mapped])
    outf = open(args.out_file, 'w') if args.out_file is not None else sys.stdout
    for preds_i in res:
        outf.write('{}\n'.format(' '.join([str(p) for p in preds_i])))


if __name__ == '__main__':
    main()
