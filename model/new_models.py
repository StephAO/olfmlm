import torch
from torch.nn import CrossEntropyLoss
from .modeling import *

class BertNSPHead(nn.Module):
    def __init__(self, config):
        super(BertNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

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
        self.nsp = BertNSPHead(config)
        self.apply(self.init_bert_weights)
        self.first_pooled_output = None
        self.config = config

    def normalize(self, x):
        return (x - torch.mean(x, 1, keepdim=True)) / torch.std(x, 1, keepdim=True)

    def forward(self, input_ids, first_pass, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, checkpoint_activations=False):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)

        lm_scores = self.lm(sequence_output)

        if first_pass:
            self.first_pooled_output = pooled_output
            return lm_scores, 0.0

        # TODO try difference, dot product, combination
        # diff = pooled_output - self.first_pooled_output
        # dot_product = torch.bmm(self.first_pooled_output.view(-1, 1, self.config.hidden_size), pooled_output.view(-1, self.config.hidden_size, 1)).view(-1, 1)
        product = self.first_pooled_output * pooled_output
        nsp_classifier_features = self.normalize(product)  #torch.cat((self.first_pooled_output, pooled_output), 1)
        nsp_scores = self.nsp(nsp_classifier_features)

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
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, checkpoint_activations=False):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)


        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class ReferentialGame(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    """
    def __init__(self, config, config_small):
        super(ReferentialGame, self).__init__(config)
        self.bert = BertModel(config)
        self.receiver = BertModel(config_small)
        self.lm = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def cosine_similarity(self, a, b):
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, checkpoint_activations=False):
        sequence_output, send_emb = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
        _, rec_emb = self.receiver(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False,
                                                   checkpoint_activations=checkpoint_activations)
        lm_scores = self.lm(sequence_output)

        cosine_similarities = self.cosine_similarity(send_emb, rec_emb)

        #print(cosine_similarities.shape)
        id_mat = torch.eye(*cosine_similarities.shape).cuda()
        bs = id_mat.shape[0]

        print('----')
        print(cosine_similarities)
        correct = id_mat * cosine_similarities / bs
        incorrect = (1. - id_mat) * cosine_similarities / (bs * (bs - 1))
        rg_loss = 2. + torch.sum(incorrect) - torch.sum(correct)
        #rg_loss = torch.clamp(1. -  id_mat * cosine_similarities + (1. - id_mat) * cosine_similarities, min=0.0)
        print(rg_loss.item(), 'c:', torch.sum(correct).item(), 'ic:', torch.sum(incorrect).item())
        #g_loss = torch.sum(rg_loss) / cosine_similarities.shape[0]

        return lm_scores, rg_loss
