import argparse
import torch

from flask import Flask
from flask import request
from flask import jsonify

from abc import ABCMeta, abstractmethod

from transquest.serving.logger import create_logger
from transquest.data.load_config import load_config
from transquest.algo.transformers.run_model import QuestModel
from transquest.data.dataset import DatasetSentLevel, DatasetWordLevel
from transquest.data.mapping_tokens_bpe import map_pieces

from sacremoses import MosesTokenizer


app = Flask(__name__)


class ModelServer(metaclass=ABCMeta):

    def __init__(self, args, logger, data_loader):
        self.args = args
        self.logger = logger
        self.config = load_config(args)
        self.model = None
        self.load_model()
        self.data_loader = data_loader
        self.target_tokenizer = MosesTokenizer(lang=args.lang_pair[-2:])
        self.source_tokenizer = MosesTokenizer(lang=args.lang_pair[:2])

    def load_model(self):
        use_cuda = False if self.args.cpu else torch.cuda.is_available()
        self.model = QuestModel(self.config['model_type'], self.args.model_dir, use_cuda=use_cuda, args=self.config)

    def predict(self, input_json):
        self.logger.info(input_json)
        try:
            test_set = self.load_data(input_json)
        except Exception:
            self.logger.exception('Exception occurred when processing input data')
            raise
        try:
            model_output = self.predict_from_model(test_set)
        except Exception:
            self.logger.exception('Exception occurred when generating predictions!')
            raise
        try:
            output = self.prepare_output(input_json, model_output)
        except Exception:
            self.logger.exception('Exception occurred when building response!')
            raise
        return output

    @staticmethod
    def tokenize(input_json, text_name, tokenizer):
        tokenized = []
        for item in input_json['data']:
            tokenized.append(tokenizer.tokenize(item[text_name]))
        return tokenized

    def load_data(self, input_json):
        test_set = self.data_loader(self.config, evaluate=True, serving_mode=True)
        test_set.make_dataset(input_json['data'])
        return test_set

    @abstractmethod
    def predict_from_model(self, test_set):
        pass

    @abstractmethod
    def prepare_output(self, input_json, model_output):
        pass


class SentenceLevelServer(ModelServer):

    def predict_from_model(self, test_set):
        _, model_output = self.model.eval_model(test_set.tensor_dataset, serving=True)
        return model_output

    def prepare_output(self, input_json, model_output):
        predictions = model_output.tolist()
        if not type(predictions) is list:
            predictions = [predictions]
        result = []
        for pred in predictions:  # return list of lists to be consistent with word-level serving
            result.append([pred])
        response = {
            'predictions': result,
            'source_tokens': self.tokenize(input_json, 'text_a', self.source_tokenizer),
            'target_tokens': self.tokenize(input_json, 'text_b', self.target_tokenizer),
        }
        self.logger.info(response)
        response = jsonify(response)
        return response


class WordLevelServer(ModelServer):

    def prepare_output(self, input_json, model_output):
        response = {
            'predictions': model_output,
            'source_tokens': self.tokenize(input_json, 'text_a', self.source_tokenizer),
            'target_tokens': self.tokenize(input_json, 'text_b', self.target_tokenizer),
        }
        return jsonify(response)

    def predict_from_model(self, test_set):
        preds = self.model.predict(test_set.tensor_dataset, serving=True)
        res = []
        for i, preds_i in enumerate(preds):
            input_ids = test_set.tensor_dataset.tensors[0][i]
            input_mask = test_set.tensor_dataset.tensors[1][i]
            preds_i = [p for j, p in enumerate(preds_i) if input_mask[j] and input_ids[j] not in (0, 2)]
            bpe_pieces = test_set.tokenizer.tokenize(test_set.examples[i].text_a)
            mt_tokens = self.target_tokenizer.tokenize(test_set.examples[i].text_a)
            mapped = map_pieces(bpe_pieces, mt_tokens, preds_i, 'average', from_sep='▁')
            res.append([int(v) for v in mapped])
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
        return model_server.predict(input_json)

    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
