import argparse
import torch

from flask import Flask
from flask import request
from flask import jsonify

from transquest.serving.logger import create_logger
from transquest.data.load_config import load_config
from transquest.algo.transformers.run_model import QuestModel
from transquest.data.dataset import DatasetSentLevel, DatasetWordLevel
from transquest.data.mapping_tokens_bpe import map_pieces


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
            model_outputs = self.get_model_predictions(test_set)
        except Exception:
            self.logger.exception('Exception occurred when generating predictions!')
            raise
        return model_outputs

    def get_model_predictions(self, test_set):
        _, model_outputs = self.model.eval_model(test_set.tensor_dataset, serving=True)
        return model_outputs


class SentenceLevelServer(ModelServer):

    def build_response(self, input_json):
        self.logger.info(input_json)
        output = self.predict(input_json)
        try:
            output = self.prepare_output(output)
            response = {'predictions': output}
            self.logger.info(response)
            response = jsonify(response)
        except Exception:
            self.logger.exception('Exception occurred when building response!')
            raise
        return response

    @staticmethod
    def prepare_output(output):
        predictions = output.tolist()
        if not type(predictions) is list:
            predictions = [predictions]
        return predictions


class WordLevelServer(ModelServer):

    def build_response(self, input_json):
        self.logger.info(input_json)
        output = self.predict(input_json)
        try:
            response = {'predictions': output}
        except Exception:
            self.logger.exception('Exception occurred when building response!')
            raise
        return response

    def get_model_predictions(self, testset):
        preds = self.model.predict(testset.tensor_dataset)
        res = []
        for i, preds_i in enumerate(preds):
            input_ids = testset.tensor_dataset.tensors[0][i]
            input_mask = testset.tensor_dataset.tensors[1][i]
            preds_i = [p for j, p in enumerate(preds_i) if input_mask[j] and input_ids[j] not in (0, 2)]
            bpe_pieces = testset.tokenizer.tokenize(testset.examples[i].text_a)
            mt_tokens = testset.examples[i].text_a.split()
            mapped = map_pieces(bpe_pieces, mt_tokens, preds_i, 'average', from_sep='▁')
            res.append([float(v) for v in mapped])
        return res


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
