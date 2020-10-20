import argparse
import torch

from flask import Flask
from flask import request
from flask import jsonify

from transquest.serving.logger import create_logger
from transquest.data.load_config import load_config
from transquest.algo.transformers.run_model import QuestModel
from transquest.data.dataset import DatasetSentLevel, DatasetWordLevel


app = Flask(__name__)


class ModelServer:

    def __init__(self, args, logger, data_loader):
        self.args = args
        self.logger = logger
        self.config = load_config(args)
        self.model = None
        self.load_model()
        self.data_loader = data_loader

    def load_model(self):
        use_cuda = False if self.args.cpu else torch.cuda.is_available()
        self.model = QuestModel(self.config['model_type'], self.args.model_dir, use_cuda=use_cuda, args=self.config)

    def predict(self, input_json):
        try:
            test_set = self.data_loader(self.config, evaluate=True, serving_mode=True)
            test_set.make_dataset(input_json['data'])
            _, model_outputs = self.model.eval_model(test_set.tensor_dataset, serving=True)
        except Exception:
            self.logger.exception('Exception occurred when generating predictions!')
            raise
        return model_outputs


class SentenceLevelServer(ModelServer):

    def build_response(self, input_json):
        self.logger.info(input_json)
        output = self.predict(input_json)
        try:
            response = self.prepare_output(output)
        except Exception:
            self.logger.exception('Exception occurred when building response!')
            raise
        return response

    def prepare_output(self, output):
        predictions = output.tolist()
        if not type(predictions) is list:
            predictions = [predictions]
        response = {'predictions': predictions}
        self.logger.info(response)
        return jsonify(response)


class WordLevelServer(ModelServer):

    def build_response(self, input_json):
        self.logger.info(input_json)
        output = self.predict(input_json)
        try:
            response = self.prepare_output(output)
        except Exception:
            self.logger.exception('Exception occurred when building response!')
            raise
        return response

    def prepare_output(self, output):
        predictions = output.tolist()
        if not type(predictions) is list:
            predictions = [predictions]
        response = {'predictions': predictions}
        self.logger.info(response)
        return jsonify(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, required=True, help='Path to the model')
    parser.add_argument('-l', '--lang_pair', type=str, required=True, help='Language pair')
    parser.add_argument('-c', '--config')
    parser.add_argument('-o', '--output_dir', default=None, required=False)
    parser.add_argument('--cpu', action='store_true', required=False, default=False)
    parser.add_argument('-p', '--port', type=int, required=True)
    parser.add_argument('--level', choices=['word', 'sentence'], required=True)
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--logging', type=str, required=False, default=None)
    args = parser.parse_args()
    logger = create_logger(path=args.logging)
    if args.level == 'sentence':
        model_server = SentenceLevelServer(args, logger, DatasetSentLevel)
    elif args.level == 'word':
        model_server = WordLevelServer(args, logger, DatasetWordLevel)
    else:
        logger.error('No QE model implemented for this prediction level. Available levels are word and sentence')
        raise NotImplementedError

    @app.route('/predict', methods=['POST'])
    def predict():
        input_json = request.json
        return model_server.build_response(input_json)

    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
