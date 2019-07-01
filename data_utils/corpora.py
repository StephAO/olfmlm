# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""several datasets with preset arguments"""
from .datasets import json_dataset, csv_dataset

class wikipedia(json_dataset):
	"""
	dataset for wikipedia with arguments configured for convenience

	command line usage: `--train-data wikipedia`
	"""
	PATH = '/scratch/gobi1/datasets/NLP-Corpus/wikipedia_version2/wikipedia_sentences.json'
	assert_str = "make sure to set PATH at line 27 of data_utils/corpora.py"
	def __init__(self, **kwargs):
		assert wikipedia.PATH != '<wikipedia_path>', \
                                         wikipedia.assert_str
		if not kwargs:
			kwargs = {}
		kwargs['text_key'] = 'text'
		kwargs['loose_json'] = True
		super(wikipedia, self).__init__(wikipedia.PATH, **kwargs)

NAMED_CORPORA = {
	'wikipedia': wikipedia,
}
