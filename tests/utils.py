import os

test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, "../data")


class Args:
    def __init__(self, config, output_dir=None, model_dir=None):
        self.config = config
        self.output_dir = output_dir
        self.model_dir = model_dir


class DataSent:

    train_classif_tsv = os.path.join(data_dir, "toy", "toy.classif.tsv")
    test_classif_tsv = os.path.join(data_dir, "toy", "toy.classif.tsv")
    train_tsv = os.path.join(data_dir, "toy", "toy.sien.tsv")
    test_tsv = os.path.join(data_dir, "toy", "toy.sien.tsv")
    config_path = os.path.join(data_dir, "toy", "toy.json")
    features_pref = os.path.join(data_dir, "toy", "features")
    out_dir = os.path.join(data_dir, "toy", "output")
    args = Args(config_path, output_dir=out_dir)


class DataWord:

    src_txt = os.path.join(data_dir, "toy-word-level", "toy.src")
    src_tags_txt = os.path.join(data_dir, "toy-word-level", "toy.source_tags")
    mt_txt = os.path.join(data_dir, "toy-word-level", "toy.mt")
    mt_tags_txt = os.path.join(data_dir, "toy-word-level", "toy.tags")
    config_path = os.path.join(data_dir, "toy-word-level", "toy.json")
    mt_raw_path = os.path.join(data_dir, "toy-word-level", "toy.mt_out")
    features_path = os.path.join(data_dir, "toy-word-level", "toy.feature1")
    out_dir = os.path.join(data_dir, "toy-word-level", "output")
    args = Args(config_path, output_dir=out_dir)


class DataWordTest(DataWord):

    args = Args(DataWord.config_path, model_dir=os.path.join(data_dir, "toy-word-level", "model"))
