import argparse
import torch

from flask import Flask
from flask import request
from flask import jsonify

from transquest.serving.logger import create_logger
from transquest.data.load_config import load_config
from transquest.algo.transformers.run_model import QuestModel
from transquest.data.dataset import DatasetSentLevel


app = Flask(__name__)


def load_model(args):
    global model
    global config
    config = load_config(args)
    use_cuda = False if args.cpu else torch.cuda.is_available()
    model = QuestModel(config['model_type'], args.model_dir, use_cuda=use_cuda, args=config)


def build_response(predictions):
    predictions = predictions.tolist()
    if not type(predictions) is list:
        predictions = [predictions]
    response = {'predictions': predictions}
    logger.info(response)
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    global model
    global logger
    global config
    input_json = request.json
    logger.info(input_json)
    try:
        test_set = DatasetSentLevel(config, evaluate=True)
        test_set.make_dataset(input_json['data'])
        result, model_outputs = model.eval_model(test_set.tensor_dataset, serving=True)
    except Exception:
        logger.exception('Exception occurred when generating predictions!')
        raise
    try:
        response = build_response(model_outputs)
    except Exception:
        logger.exception('Exception occurred when building response!')
        raise
    return response


def main():
    global model
    global config
    global logger
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, required=True, help='Path to the model')
    parser.add_argument('-l', '--lang_pair', type=str, required=True, help='Language pair')
    parser.add_argument('-c', '--config')
    parser.add_argument('-o', '--output_dir', default=None, required=False)
    parser.add_argument('--cpu', action='store_true', required=False, default=False)
    parser.add_argument('-p', '--port', type=int, required=True)
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--logging', type=str, required=False, default=None)
    args = parser.parse_args()
    logger = create_logger(path=args.logging)
    load_model(args)
    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
