import os
import json

from multiprocessing import cpu_count


def load_config(cli_args):
    config = json.load(open(cli_args.config))
    try:
        if cli_args.model_dir is not None:
            assert cli_args.output_dir is None
            config['model_name'] = cli_args.model_dir  # for prediction
    except AttributeError:
        pass
    try:
        if cli_args.output_dir is not None:
            config.update({  # for training
                'output_dir': os.path.join(cli_args.output_dir, 'outputs'),
                'best_model_dir': os.path.join(cli_args.output_dir, 'best_model'),
                'cache_dir': os.path.join(cli_args.output_dir, 'cache_dir'),
            })
    except AttributeError:
        pass
    return config
