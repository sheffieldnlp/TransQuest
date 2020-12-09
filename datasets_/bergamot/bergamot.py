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
"""Bergamot Datasets"""

import os

import logging

import datasets


_CITATION = """
"""

_DESCRIPTION = """\
For details see
"""


class PtakopetConfig(datasets.BuilderConfig):
    """BuilderConfig for a dataset in the Bergamot project"""

    def __init__(self, **kwargs):
        """BuilderConfig for a dataset in the Bergamot project.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PtakopetConfig, self).__init__(**kwargs)


class Bergamot(datasets.GeneratorBasedBuilder):
    """ dataset."""

    BUILDER_CONFIGS = [
        PtakopetConfig(name="bergamot", version=datasets.Version("1.0.0"), description="Bergamot"),
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
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if not self.config.data_dir:
            raise ValueError(f"Must specify the folder where the files are, but got data_dir={self.config.data_dir}")
        data_dir = self.config.data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "src_path": os.path.join(data_dir, "test.src"),
                    "mt_path": os.path.join(data_dir, "test.mt"),
                    "mt_tags_path": os.path.join(data_dir, "test.word_tags"),
                },
            ),
        ]

    def _generate_examples(self, src_path, mt_path, mt_tags_path):
        logging.info("Generating examples")
        with open(src_path, encoding="utf-8") as src_file, open(mt_path, encoding="utf-8") as mt_file, open(
            mt_tags_path, encoding="utf-8"
        ) as mt_tags_file:
            for id, (src, mt, mt_tags) in enumerate(zip(src_file, mt_file, mt_tags_file)):
                raw_mt_tags = mt_tags.strip().split()
                mt_tags = ["OK" if tag == "0" else "BAD" for tag in raw_mt_tags]
                src_tokens = src.strip().split()
                yield id, {
                    "src": src_tokens,
                    "mt": mt.strip().split(),
                    "src_tags": ["OK"] * len(src_tokens),
                    "mt_tags": mt_tags,
                }
