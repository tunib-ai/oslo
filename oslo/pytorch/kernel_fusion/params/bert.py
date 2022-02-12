import torch

from oslo.pytorch.kernel_fusion.params import Params, register_params, TensorMeta


class BertParams(Params):
    @staticmethod
    def supported_args():
        return [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
            "labels",
            "output_hidden_states",
            "use_cache",
            "return_dict",
            "next_sentence_label",
        ]

    @register_params(module_cls="BertEmbeddings")
    def bert_embeddings(self):
        # from transformers.models.bert.modeling_bert import BertEmbeddings

        return {
            "input_ids": [
                TensorMeta(self.bsz, self.seq_len, dtype=torch.long, necessary=True),
            ],
            "token_type_ids": [
                TensorMeta(self.bsz, self.seq_len, dtype=torch.long),
            ],
            "position_ids": [
                TensorMeta(self.bsz, self.seq_len, dtype=torch.long),
            ],
        }

    @register_params(module_cls="BertLayer")
    def bert_layer(self):
        # from transformers.models.bert.modeling_bert import BertLayer
        return {
            "hidden_states": [
                TensorMeta(self.bsz, self.seq_len, self.hid_size, necessary=True),
            ],
            "attention_mask": [
                TensorMeta(self.bsz, 1, 1, self.seq_len),
            ],
        }

    @register_params(module_cls="BertPooler", model_cls="BertModel")
    def bert_pooler(self):
        # from transformers.models.bert.modeling_bert import BertPooler
        return {
            "hidden_states": [
                TensorMeta(self.bsz, self.seq_len, self.hid_size, necessary=True),
            ],
        }

    @register_params(module_cls="BertPreTrainingHeads")
    def bert_pretraining_heads(self):
        # from transformers.models.bert.modeling_bert import BertPreTrainingHeads
        return {
            "sequence_output": [
                TensorMeta(self.bsz, self.seq_len, self.hid_size, necessary=True),
            ],
            "pooled_output": [
                TensorMeta(self.bsz, self.hid_size, necessary=True),
            ],
        }

    @register_params(module_cls="BertOnlyMLMHead", model_cls="BertLMHeadModel")
    def bert_only_mlm_head_for_clm(self):
        # from transformers.models.bert.modeling_bert import BertOnlyMLMHead
        return {
            "sequence_output": [
                TensorMeta(self.bsz, self.seq_len, self.hid_size, necessary=True),
            ],
        }

    @register_params(module_cls="BertOnlyMLMHead", model_cls="BertForMaskedLM")
    def bert_only_mlm_head_for_mlm(self):
        # from transformers.models.bert.modeling_bert import BertOnlyMLMHead
        return {
            "sequence_output": [
                TensorMeta(self.bsz, self.seq_len, self.hid_size, necessary=True),
            ],
        }

    @register_params(
        module_cls="BertOnlyNSPHead", model_cls="BertForNextSentencePrediction"
    )
    def bert_only_nsp_head(self):
        # from transformers.models.bert.modeling_bert import BertOnlyNSPHead
        return {
            "pooled_output": [
                TensorMeta(self.bsz, self.hid_size, necessary=True),
            ],
        }
