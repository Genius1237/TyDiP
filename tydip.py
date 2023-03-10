# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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
"""TyDiP: A Multilingual Politeness Dataset"""


import csv
from dataclasses import dataclass
import datasets
from datasets.tasks import TextClassification


_CITATION = """\
@inproceedings{srinivasan-choi-2022-tydip,
    title = "{T}y{D}i{P}: A Dataset for Politeness Classification in Nine Typologically Diverse Languages",
    author = "Srinivasan, Anirudh  and
      Choi, Eunsol",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.420",
    pages = "5723--5738",
}"""

_DESCRIPTION = """\
The TyDiP dataset is a dataset of requests in conversations between wikipedia editors
that have been annotated for politeness. The splits available below consists of only
requests from the top 25 percentile (polite) and bottom 25 percentile (impolite) of
politeness scores. The English train set and English test set that are
adapted from the Stanford Politeness Corpus, and test data in 9 more languages
(Hindi, Korean, Spanish, Tamil, French, Vietnamese, Russian, Afrikaans, Hungarian) 
was annotated by us.
"""

_LANGUAGES = ("en", "hi", "ko", "es", "ta", "fr", "vi", "ru", "af", "hu")


# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
# _URL = "https://huggingface.co/datasets/Genius1237/TyDiP/resolve/main/data/binary/"
_URL = "https://huggingface.co/datasets/Genius1237/TyDiP/raw/main/data/binary/"
_URLS = {
    'en': {
        'train': _URL + 'en_train_binary.csv',
        'test': _URL + 'en_test_binary.csv'
    },
} | {lang: {'test': _URL + '{}_test_binary.csv'.format(lang)} for lang in _LANGUAGES[1:]}


@dataclass
class TyDiPConfig(datasets.BuilderConfig):
    """BuilderConfig for TyDiP."""
    lang: str = None


class MultilingualLibrispeech(datasets.GeneratorBasedBuilder):
    """TyDiP dataset."""

    BUILDER_CONFIGS = [
        TyDiPConfig(name=lang, lang=lang) for lang in _LANGUAGES
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "labels": datasets.ClassLabel(num_classes=2, names=[0, 1]),
                }
            ),
            supervised_keys=("text", "labels"),
            homepage=_URL,
            citation=_CITATION,
            task_templates=[TextClassification(text_column="text", label_column="labels")],
        )

    def _split_generators(self, dl_manager):
        splits = []
        if self.config.lang == 'en':
            file_path = dl_manager.download_and_extract(_URLS['en']['train'])
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN, gen_kwargs={"data_file": file_path}
                ))
        file_path = dl_manager.download_and_extract(_URLS[self.config.lang]['test'])
        splits.append(
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"data_file": file_path}
            )
        )
        return splits

    def _generate_examples(self, data_file):
        """Generate examples from a TyDiP data file"""
        with open(data_file) as f:
            csv_reader = csv.reader(f)
            for i, row in enumerate(csv_reader):
                if i != 0:
                    yield i - 1, {
                        "text": row[0],
                        "labels": int(float(row[1]) > 0),
                    }
