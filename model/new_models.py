import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from sentence_encoders.model.modeling import *

class BertNSPHead(nn.Module):
    def __init__(self, config, num_classes=2):
        super(BertNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, num_classes)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class Split(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.
    """
    def __init__(self, config):
        super(Split, self).__init__(config)
        self.bert = BertModel(config)
        self.lm = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.corrupted = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)
        self.first_pooled_output = None
        self.config = config

    def normalize(self, x):
        return (x - torch.mean(x, 1, keepdim=True)) / torch.std(x, 1, keepdim=True)

    def forward(self, input_ids, first_pass, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, checkpoint_activations=False):
        sequence_output = []
        pooled_output = []
        for i in range(2):
            s_o, p_o = self.bert(input_ids[i], None, attention_mask[i],
                                 output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
            sequence_output.append(s_o)
            pooled_output.append(p_o)

        lm_scores = torch.cat((self.lm(sequence_output[0]), self.lm(sequence_output[1])), dim=0)

        # TODO try difference, dot product, combination
        # diff = pooled_output - self.first_pooled_output
        # dot_product = torch.bmm(self.first_pooled_output.view(-1, 1, self.config.hidden_size), pooled_output.view(-1, self.config.hidden_size, 1)).view(-1, 1)
        product = pooled_output[0] * pooled_output[1]
        nsp_classifier_features = self.normalize(product)  #torch.cat((self.first_pooled_output, pooled_output), 1)
        nsp_scores = self.corrupted(nsp_classifier_features)

        return lm_scores, nsp_scores


class Corrupt(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    """
    def __init__(self, config):
        super(Corrupt, self).__init__(config)
        self.bert = BertModel(config)
        self.lm = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.corrupted = BertNSPHead(config, num_classes=2)
        # self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, checkpoint_activations=False):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
        #prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        lm_scores = self.lm(sequence_output)
        corrupted_scores = self.corrupted(pooled_output)

        return lm_scores, corrupted_scores


class ReferentialGame(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    """
    def __init__(self, config):
        super(ReferentialGame, self).__init__(config)
        self.bert = BertModel(config)
        # self.receiver = BertModel(config_small)
        self.lm = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def cosine_similarity(self, a, b):
        "taken from https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re"
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def inner_product(self, a, b):
        return torch.mm(a, b.transpose(0, 1))

    def mse(self, a, b):
        '''
        taken from: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
        '''
        a_norm = (a ** 2).sum(1).view(-1, 1)
        b_t = torch.transpose(b, 0, 1)
        b_norm = (b ** 2).sum(1).view(1, -1)
        dist = a_norm + b_norm - 2.0 * torch.mm(a, b_t)

        return torch.clamp(dist, 0.0, np.inf)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, checkpoint_activations=False):
        seq_output, pooled_output = self.bert(torch.cat(input_ids, dim=0), None, torch.cat(attention_mask, dim=0),
                                              output_all_encoded_layers=False,
                                              checkpoint_activations=checkpoint_activations)
        half = len(input_ids[0])
        send_emb, recv_emb = pooled_output[:half], pooled_output[half:]
        lm_scores = self.lm(seq_output)
        rg_scores = self.mse(send_emb, recv_emb)

        return lm_scores, rg_scores

class Combined(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    """
    def __init__(self, config):
        super(Combined, self).__init__(config)
        self.bert = BertModel(config)
        self.lm = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.corrupted = BertNSPHead(config, num_classes=2)
        self.apply(self.init_bert_weights)

    def cosine_similarity(self, a, b):
        "taken from https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re"
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def inner_product(self, a, b):
        return torch.mm(a, b.transpose(0, 1))


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, checkpoint_activations=False):
        seq_output, pooled_output = self.bert(torch.cat(input_ids, dim=0), None, torch.cat(attention_mask, dim=0),
                                              output_all_encoded_layers=False,
                                              checkpoint_activations=checkpoint_activations)
        
        half = len(input_ids[0])
        send_emb, recv_emb = pooled_output[:half], pooled_output[half:]
        lm_scores = self.lm(seq_output)
        rg_scores = self.cosine_similarity(send_emb, recv_emb)
        corrupted_scores = self.corrupted(pooled_output)
        return lm_scores, rg_scores, corrupted_scores
