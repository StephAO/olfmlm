import logging as log
from typing import Dict

import torch
import torch.nn as nn
from allennlp.modules import scalar_mix

# huggingface implementation of BERT
from ..model.modeling import BertModel
from ..model.modeling import BertConfig

from ..data_utils.wordpiece import BertTokenizer

from .preprocess import parse_task_list_arg


def _get_seg_ids(ids, sep_id):
    """ Dynamically build the segment IDs for a concatenated pair of sentences
    Searches for index SEP_ID in the tensor

    args:
        ids (torch.LongTensor): batch of token IDs

    returns:
        seg_ids (torch.LongTensor): batch of segment IDs

    example:
    > sents = ["[CLS]", "I", "am", "a", "cat", ".", "[SEP]", "You", "like", "cats", "?", "[SEP]"]
    > token_tensor = torch.Tensor([[vocab[w] for w in sent]]) # a tensor of token indices
    > seg_ids = _get_seg_ids(token_tensor, sep_id=102) # BERT [SEP] ID
    > assert seg_ids == torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    """
    sep_idxs = (ids == sep_id).nonzero()[:, 1]
    seg_ids = torch.ones_like(ids)
    for row, idx in zip(seg_ids, sep_idxs[::2]):
        row[: idx + 1].fill_(0)
    return seg_ids

def _split_sentence(ids, cls_id, sep_id, pad_id):
    """ Splits pairs of sentences into two sentences
    Searches for index SEP_ID in the tensor

    args:
        ids (torch.LongTensor): batch of token IDs

    returns:
        seg_ids (torch.LongTensor): batch of segment IDs

    example:
    > sents = ["[CLS]", "I", "am", "a", "cat", ".", "[SEP]", "You", "like", "cats", "?", "[SEP]"]
    > token_tensor = torch.Tensor([[vocab[w] for w in sent]]) # a tensor of token indices
    > seg_ids = _get_seg_ids(token_tensor, sep_id=102) # BERT [SEP] ID
    > assert seg_ids == torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    """
    sep_idxs = (ids == sep_id).nonzero()[:, 1]
    if sep_idxs.shape[0] == ids.shape[0]:
        return ids, None
    sent_1, sent_2 = torch.full_like(ids, pad_id), torch.full_like(ids, pad_id)
    for i, (row, idx_1, idx_2) in enumerate(zip(ids, sep_idxs[::2], sep_idxs[1::2])):
        sent_1[i, :idx_1 + 1] = row[:idx_1 + 1]
        sent_2[i, 0] = cls_id
        sent_2[i, 1:(idx_2 - idx_1) + 1] = row[idx_1 + 1:idx_2 + 1]
    return sent_1, sent_2


class BertEmbedderModule(nn.Module):
    """ Wrapper for BERT module to fit into jiant APIs. """

    def __init__(self, args, cache_dir=None):
        super(BertEmbedderModule, self).__init__()

        self.two_models = args.two_models

        if args.bert_use_pretrain:
            self.model = BertModel.from_pretrained(
                args.input_module, cache_dir=cache_dir
            )
        else:
            self.config = BertConfig(args.bert_config_file)
            self.model = BertModel(self.config)
            if args.two_models:
                self.model_2 = BertModel(self.config)
        self.embeddings_mode = args.bert_embeddings_mode

        tokenizer = BertTokenizer.from_pretrained(
            args.input_module, cache_dir=cache_dir
        )

        self.split = args.split

        self._cls_id = tokenizer.vocab["[CLS]"]
        self._sep_id = tokenizer.vocab["[SEP]"]
        self._pad_id = tokenizer.vocab["[PAD]"]

        # Set trainability of this module.
        for param in self.model.parameters():
            param.requires_grad = bool(args.transfer_paradigm == "finetune")

        # Configure scalar mixing, ELMo-style.
        if self.embeddings_mode == "mix":
            if args.transfer_paradigm == "frozen":
                log.warning(
                    "NOTE: bert_embeddings_mode='mix', so scalar "
                    "mixing weights will be fine-tuned even if BERT "
                    "model is frozen."
                )
            # TODO: if doing multiple target tasks, allow for multiple sets of
            # scalars. See the ELMo implementation here:
            # https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L115
            assert len(parse_task_list_arg(args.target_tasks)) <= 1, (
                "bert_embeddings_mode='mix' only supports a single set of "
                "scalars (but if you need this feature, see the TODO in "
                "the code!)"
            )
            num_layers = self.model.config.num_hidden_layers
            self.scalar_mix = scalar_mix.ScalarMix(num_layers + 1, do_layer_norm=False)

    def normalize(self, x):
        return (x - torch.mean(x, 1, keepdim=True)) / torch.std(x, 1, keepdim=True)

    def forward(
        self, sent: Dict[str, torch.LongTensor], unused_task_name: str = "", is_pair_task=False
    ) -> torch.FloatTensor:
        """ Run BERT to get hidden states.

        This forward method does preprocessing on the go,
        changing token IDs from preprocessed bert to
        what AllenNLP indexes.

        Args:
            sent: batch dictionary
            is_pair_task (bool): true if input is a batch from a pair task

        Returns:
            h: [batch_size, seq_len, d_emb]
        """
        assert "bert_wpm_pretokenized" in sent
        # <int32> [batch_size, var_seq_len]
        ids = sent["bert_wpm_pretokenized"]
        # BERT supports up to 512 tokens; see section 3.2 of https://arxiv.org/pdf/1810.04805.pdf
        assert ids.size()[1] <= 512

        mask = ids != 0
        # "Correct" ids to account for different indexing between BERT and
        # AllenNLP.
        # The AllenNLP indexer adds a '@@UNKNOWN@@' token to the
        # beginning of the vocabulary, *and* treats that as index 1 (index 0 is
        # reserved for padding).
        ids[ids == 0] = self._pad_id + 2  # Shift the indices that were at 0 to become 2.
        # Index 1 should never be used since the BERT WPM uses its own
        # unk token, and handles this at the string level before indexing.
        assert (ids > 1).all()
        ids -= 2  # shift indices to match BERT wordpiece embeddings

        if self.embeddings_mode not in ["none", "top"]:
            # This is redundant with the lookup inside BertModel,
            # but doing so this way avoids the need to modify the BertModel
            # code.
            # Extract lexical embeddings; see
            # https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L186  # noqa
            h_lex = self.model.embeddings.word_embeddings(ids)
            h_lex = self.model.embeddings.LayerNorm(h_lex)
            # following our use of the OpenAI model, don't use dropout for
            # probing. If you would like to use dropout, consider applying
            # later on in the SentenceEncoder (see models.py).
            #  h_lex = self.model.embeddings.dropout(embeddings)
        if self.embeddings_mode != "only":
            # encoded_layers is a list of layer activations, each of which is
            # <float32> [batch_size, seq_len, output_dim]
            if self.split:
                s1, s2 = _split_sentence(ids, self._cls_id, self._sep_id, self._pad_id)
                token_types = torch.zeros_like(ids)

                u = self.model(s1, token_type_ids=token_types, attention_mask=mask, output_all_encoded_layers=False)[0]
                v = None if s2 is None else \
                    self.model(s2, token_type_ids=token_types, attention_mask=mask, output_all_encoded_layers=False)[0]

                if self.two_models:
                    u2 = self.model_2(s1, token_type_ids=token_types, attention_mask=mask, output_all_encoded_layers=False)[0]
                    v2 = None if s2 is None else \
                        self.model_2(s2, token_type_ids=token_types, attention_mask=mask, output_all_encoded_layers=False)[0]
                    u = torch.cat([u, u2], dim=2)
                    v = None if v is None else torch.cat([v, v2], dim=2)

                h_enc = self.normalize(u * v) if v is not None else u

            else:
                token_types = _get_seg_ids(ids, self._sep_id) if is_pair_task else torch.zeros_like(ids)
                u, _ = self.model(ids, token_type_ids=token_types, attention_mask=mask, output_all_encoded_layers=False)
                h_enc = u

        if self.embeddings_mode in ["none", "top"]:
            h = h_enc
        elif self.embeddings_mode == "only":
            h = h_lex
        elif self.embeddings_mode == "cat":
            h = torch.cat([h_enc, h_lex], dim=2)
        elif self.embeddings_mode == "mix":
            log.error("Oops, this mode has been disable, go code the fix")
            exit(0)
            # h = self.scalar_mix([h_lex] + encoded_layers, mask=mask)
        else:
            raise NotImplementedError(f"embeddings_mode={self.embeddings_mode}" " not supported.")

        # <float32> [batch_size, var_seq_len, output_dim]
        return h

    def get_output_dim(self):
        if self.embeddings_mode == "cat" or self.two_models:
            return 2 * self.model.config.hidden_size
        else:
            return self.model.config.hidden_size
