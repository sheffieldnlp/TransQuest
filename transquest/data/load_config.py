import os
import json

from multiprocessing import cpu_count


def load_config(args):
    config = json.load(open(args.config))
    process_count = cpu_count() - 2 if cpu_count() > 2 else 1
    if args.output_dir is not None:
        config.update({
            'output_dir': os.path.join(args.output_dir, 'outputs'),
            'best_model_dir': os.path.join(args.output_dir, 'best_model'),
            'cache_dir': os.path.join(args.output_dir, 'cache_dir'),
        })
    config.update({
        'process_count': process_count,
    })
    return config
