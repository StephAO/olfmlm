import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from olfmlm.model.modeling import *

class BertSentHead(nn.Module):
    def __init__(self, config, num_classes=2):
        super(BertSentHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, num_classes)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertHeadTransform(nn.Module):
    def __init__(self, config, input_size=None):
        super(BertHeadTransform, self).__init__()
        input_size = input_size if input_size else config.hidden_size
        self.dense = nn.Linear(input_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMTokenHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, input_size=None):
        super(BertLMTokenHead, self).__init__()
        self.transform = BertHeadTransform(config, input_size=input_size)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertTokenHead(nn.Module):
    def __init__(self, config, num_classes=2, input_size=None):
        super(BertTokenHead, self).__init__()
        input_size = input_size if input_size else config.hidden_size
        self.transform = BertHeadTransform(config, input_size=input_size)
        self.decoder = nn.Linear(config.hidden_size, num_classes)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        predictions = self.decoder(hidden_states)
        return predictions


class Bert(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, modes=["mlm"]):
        super(Bert, self).__init__(config)
        self.bert = BertModel(config)
        self.lm = BertLMTokenHead(config, self.bert.embeddings.word_embeddings.weight)
        self.sent = torch.nn.ModuleDict()
        self.tok = torch.nn.ModuleDict()
        if "nsp" in modes:
            self.sent["nsp"] = BertSentHead(config, num_classes=2)
        if "psp" in modes:
            self.sent["psp"] = BertSentHead(config, num_classes=3)
        if "sd" in modes:
            self.sent["sd"] = BertSentHead(config, num_classes=3)
        if "so" in modes:
            self.sent["so"] = BertSentHead(config, num_classes=2)
        if "sc" in modes:
            self.sent["sc"] = BertSentHead(config, num_classes=2)
        if "sbo" in modes:
            self.tok["sbo"] = BertLMTokenHead(config, self.bert.embeddings.word_embeddings.weight,
                                              input_size=config.hidden_size * 2)
        if "cap" in modes:
            self.tok["cap"] = BertTokenHead(config, num_classes=2)
        if "wlen" in modes:
            self.tok["wlen"] = BertTokenHead(config, num_classes=1)
        if "tf" in modes:
            self.tok["tf"] = BertTokenHead(config, num_classes=1)
        if "tf_idf" in modes:
            self.tok["tf_idf"] = BertTokenHead(config, num_classes=1)
        if "tc" in modes:
            self.tok["tc"] = BertTokenHead(config, num_classes=2)
        if "rg" in modes:
            self.sent["rg"] = BertHeadTransform(config)
        if "fs" in modes:
            self.sent["fs"] = BertHeadTransform(config)
            self.tok["fs"] = BertHeadTransform(config)
        if "tgs" in modes:
            self.tok["tgs"] = BertTokenHead(config, num_classes=6, input_size=config.hidden_size * 3)
        self.apply(self.init_bert_weights)

    def forward(self, modes, input_ids, token_type_ids=None, task_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, checkpoint_activations=False):
        # assert len(input_ids) * len(token_type_ids) * len(attention_mask) == 1
        token_type_ids = token_type_ids if token_type_ids is None else torch.cat(token_type_ids, dim=0)
        task_ids = task_ids if task_ids is None else torch.cat(task_ids, dim=0)
        att_mask = attention_mask if attention_mask is None else torch.cat(attention_mask, dim=0)
        sequence_output, pooled_output = self.bert(torch.cat(input_ids, dim=0), token_type_ids, task_ids, att_mask,
                                                   output_all_encoded_layers=False,
                                                   checkpoint_activations=checkpoint_activations)

        scores = {}
        if "mlm" in modes:
            scores["mlm"] = self.lm(sequence_output)
        if "nsp" in modes:
            scores["nsp"] = self.sent["nsp"](pooled_output)
        if "psp" in modes:
            scores["psp"] = self.sent["psp"](pooled_output)
        if "sd" in modes:
            scores["sd"] = self.sent["sd"](pooled_output)
        if "so" in modes:
            scores["so"] = self.sent["so"](pooled_output)
        if "rg" in modes:
            half = len(input_ids[0])
            send_emb, recv_emb = pooled_output[:half], pooled_output[half:]
            send_emb, recv_emb = self.sent["rg"](send_emb), self.sent["rg"](recv_emb)
            scores["rg"] = self.cosine_similarity(send_emb, recv_emb)
        if "fs" in modes:
            half = len(input_ids[0])
            prev_emb, next_emb = pooled_output[:half], pooled_output[half:]
            prev_emb, next_emb = self.sent["fs"](prev_emb), self.sent["fs"](next_emb)
            prev_words, next_words = sequence_output[:half], sequence_output[half:]
            prev_words, next_words = self.tok["fs"](prev_words), self.tok["fs"](next_words)
            s1 = self.batch_cos_sim(next_words, prev_emb) #torch.torch.sigmoid(torch.bmm(next_words, prev_emb[:, :, None]))
            s2 = self.batch_cos_sim(prev_words, next_emb) #torch.sigmoid(torch.bmm(prev_words, next_emb[:, :, None]))
            sim = torch.cat((s1, s2), dim=1).squeeze().view(-1)
            #ref = torch.zeros_like(sim)
            scores["fs"] = sim #torch.stack((ref, sim), dim=1)
        if "sbo" in modes:
            output_concats = [torch.cat((sequence_output[:, 0], sequence_output[:, 0], sequence_output[:, 0]), dim=-1)]
            output_concats += [torch.cat((sequence_output[:, 0], sequence_output[:, 0], sequence_output[:, 1]), dim=-1)]
            for i in range(2, sequence_output.shape[1]):
                output_concats += [torch.cat((sequence_output[:, i - 2], sequence_output[:, i - 1],
                                              sequence_output[:, i]), dim=-1)]
            output_concats += [torch.cat((sequence_output[:, i + 2], sequence_output[:, i + 2]), dim=-1)]
            output_concats = torch.stack(output_concats, dim=1)
            scores["sbo"] = self.tok["sbo"](output_concats)
        if "cap" in modes:
            scores["cap"] = self.tok["cap"](sequence_output)
        if "wlen" in modes:
            scores["wlen"] = self.tok["wlen"](sequence_output)
        if "tf" in modes:
            scores["tf"] = self.tok["tf"](sequence_output)
        if "tf_idf" in modes:
            scores["tf_idf"] = self.tok["tf_idf"](sequence_output)
        if "sc" in modes:
            scores["sc"] = self.sent["sc"](pooled_output)
        if "tc" in modes:
            scores["tc"] = self.tok["tc"](sequence_output)
        if "tgs" in modes:
            output_concats = [torch.cat((sequence_output[:, 0], sequence_output[:, 0]), dim=-1)]
            # output_concats += [torch.cat((sequence_output[:, 0], sequence_output[:, 0], sequence_output[:, 1]), dim=-1)]
            for i in range(1, sequence_output.shape[1]):
                output_concats += [torch.cat((sequence_output[:, i - 1], sequence_output[:, i]), dim=-1)]
            output_concats = torch.stack(output_concats, dim=1)
            scores["tgs"] = self.tok["tgs"](output_concats)

        return scores

    def cosine_similarity(self, a, b):
        "taken from https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re"
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def batch_cos_sim(self, a, b):
        a_norm = a / a.norm(dim=2)[:, :, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return torch.bmm(a_norm, b_norm[:, :, None])

    def inner_product(self, a, b):
        return torch.mm(a, b.transpose(0, 1))
