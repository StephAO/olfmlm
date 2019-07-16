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

"""Utilities for wrapping BertModel."""

import torch

from .modeling import BertConfig
from .modeling import BertForPreTraining as Bert
from .modeling import BertLayerNorm

from .new_models import Split, Corrupt

sentence_encoders = {
    "bert" : Bert,
    "split" : Split,
    "corrupt": Corrupt
}

def get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0}
    for module_ in module.modules():
        if isinstance(module_, (BertLayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params


class BertModel(torch.nn.Module):

    def __init__(self, tokenizer, args):

        super(BertModel, self).__init__()
        # if args.pretrained_bert:
        #     self.model = BertForPreTraining.from_pretrained(
        #         args.tokenizer_model_type,
        #         cache_dir=args.cache_dir,
        #         fp32_layernorm=args.fp32_layernorm,
        #         fp32_embedding=args.fp32_embedding,
        #         layernorm_epsilon=args.layernorm_epsilon)
        # else:
        if args.bert_config_file is None:
            raise ValueError("If not using a pretrained_bert, please specify a bert config file")
        self.config = BertConfig(args.bert_config_file)
        self.model = sentence_encoders[args.model_type](self.config) # TODO change to args
        self.model_type = args.model_type

    def forward(self, input_tokens, token_type_ids=None, attention_mask=None, checkpoint_activations=False, first_pass=False):
        if self.model_type != "split":
            r = self.model(input_tokens, token_type_ids, attention_mask, checkpoint_activations=checkpoint_activations)
        else:
            r = self.model(input_tokens, first_pass, token_type_ids, attention_mask, checkpoint_activations=checkpoint_activations)
        return r

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix,
                                     keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def get_params(self):
        param_groups = []
        param_groups += list(get_params_for_weight_decay_optimization(self.model.bert.encoder.layer))
        param_groups += list(get_params_for_weight_decay_optimization(self.model.bert.pooler))
        param_groups += list(get_params_for_weight_decay_optimization(self.model.bert.embeddings))
        if self.model_type == "split":
            param_groups += list(get_params_for_weight_decay_optimization(self.model.nsp.seq_relationship))
            param_groups += list(get_params_for_weight_decay_optimization(self.model.lm.predictions.transform))
            param_groups[1]['params'].append(self.model.lm.predictions.bias)
        else:
            param_groups += list(get_params_for_weight_decay_optimization(self.model.cls.seq_relationship))
            param_groups += list(get_params_for_weight_decay_optimization(self.model.cls.predictions.transform))
            param_groups[1]['params'].append(self.model.cls.predictions.bias)

        return param_groups
