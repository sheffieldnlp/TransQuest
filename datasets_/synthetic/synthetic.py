# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""WMT2020: Quality Estimation Task"""

import os

import logging

import datasets


_CITATION = """"""

_DESCRIPTION = """"""


class SyntheticSamplingConfig(datasets.BuilderConfig):
    """BuilderConfig for WMT2020QE"""

    def __init__(self, **kwargs):
        """BuilderConfig for WMT2020 QE.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SyntheticSamplingConfig, self).__init__(**kwargs)


class SyntheticSampling(datasets.GeneratorBasedBuilder):
    """ dataset."""

    BUILDER_CONFIGS = [
        SyntheticSamplingConfig(
            name="synthetic", version=datasets.Version("1.0.0"), description="WMT2020 Quality Estimation dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "src": datasets.Sequence(datasets.Value("string")),
                    "mt": datasets.Sequence(datasets.Value("string")),
                    "src_tags": datasets.Sequence(datasets.features.ClassLabel(names=["OK", "BAD"])),
                    "mt_tags": datasets.Sequence(datasets.features.ClassLabel(names=["OK", "BAD"])),
                    "hter": datasets.Value("float32"),
                }
            ),
            supervised_keys=None,
            homepage="http://www.statmt.org/wmt20/quality-estimation-task.html",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if not self.config.data_dir:
            raise ValueError(f"Must specify the folder where the files are, but got data_dir={self.config.data_dir}")
        data_dir = self.config.data_dir
        generators = []
        generators.append(
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "src_path": os.path.join(data_dir, "train.src"),
                    "mt_path": os.path.join(data_dir, "train.mt"),
                    "mt_tags_path": os.path.join(data_dir, "train.tags"),
                    "hter_path": os.path.join(data_dir, "train.hter"),
                },
            )
        )

        return generators

    def _generate_examples(self, src_path, mt_path, mt_tags_path, hter_path):
        logging.info("Generating examples")
        with open(src_path, encoding="utf-8") as src_file, open(mt_path, encoding="utf-8") as mt_file, open(
                mt_tags_path, encoding="utf-8") as mt_tags_file, open(
            hter_path, encoding="utf-8"
        ) as hter_file:
            for id, (src, mt, mt_tags, hter) in enumerate(
                zip(src_file, mt_file, mt_tags_file, hter_file)
            ):
                src_tokens = src.strip().split()
                yield id, {
                    "src": src_tokens,
                    "mt": mt.strip().split(),
                    "src_tags": ["OK"] * len(src_tokens),
                    "mt_tags": mt_tags.strip().split(),
                    "hter": float(hter.strip()),
                }
