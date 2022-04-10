from transformers.models.gpt2.modeling_gpt2 import *

from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from colossalai.nn.layer.parallel_sequence import RingQK, RingAV


config = {
    'parallel' : dict(
        pipeline=1,
        tensor=dict(size=4, mode='sequence')
    )
}
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
host = os.environ['MASTER_ADDR']
port = int(os.environ['MASTER_PORT'])
backend = 'nccl'
gpc.load_config(config)
gpc.init_global_dist(rank, world_size, backend, host, port)
gpc.init_parallel_groups()

# set cuda device
if torch.cuda.is_available():
    # if local rank is not given, calculate automatically
    gpc.set_device(local_rank)


class GPT2WithSPLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2WithSPModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()


class GPT2WithSPModel(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2WithSPBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self._split_inputs(input_ids, token_type_ids, position_ids)
        # super().forward(
        #     input_ids,
        #     past_key_values,
        #     attention_mask,
        #     token_type_ids,
        #     position_ids,
        #     head_mask,
        #     inputs_embeds,
        #     encoder_hidden_states,
        #     encoder_attention_mask,
        #     use_cache,
        #     output_attentions,
        #     output_hidden_states,
        #     return_dict,
        # )

    def _split_inputs(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None
    ):
        seq_length = input_ids.size(1)
        local_rank = gpc.get_local_rank(ParallelMode.SEQUENCE)
        local_world_size = gpc.get_world_size(ParallelMode.SEQUENCE)
        sub_seq_length = seq_length // local_world_size
        start_idx = local_rank * sub_seq_length
        end_idx = min((local_rank+1) * sub_seq_length, seq_length)
        if input_ids is not None:
            input_ids = input_ids[:, start_idx:end_idx].contiguous().to(torch.cuda.current_device())
            print(input_ids.size())
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, start_idx: end_idx].contiguous().to(torch.cuda.current_device())
            print(token_type_ids.size())
        if position_ids is not None:
            position_ids = position_ids[:, start_idx: end_idx].contiguous().to(torch.cuda.current_device())
            print(position_ids.size())
        else:
            position_ids = torch.arange(start_idx, end_idx, dtype=torch.long, device=torch.cuda.current_device())
            position_ids = position_ids.unsqueeze(0).view(-1, end_idx - start_idx)

        return input_ids, token_type_ids, position_ids

class GPT2WithSPBlock(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)


class GPT2RingAttention(GPT2Attention):
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        if self.reorder_and_upcast_attn:
            raise NotImplementedError(
                "Not implemented for Sequence Parallelism with `reorder_and_upcast_attn=True`"
            )
        batch_size = query.size(0)
        head_count = query.size(1)
        sub_seq_length = query.size(2)
        query = query.view(batch_size * head_count, sub_seq_length, -1)
        key = key.view(batch_size * head_count, sub_seq_length, -1)
        attn_weights = RingQK.apply(
            query.contiguous(),  # [batch_size * num_heads, sub_seq_len, head_size]
            key.contiguous(),  # [batch_size * num_heads, sub_seq_len, head_size],
            batch_size,
            self.num_heads,
            sub_seq_length
        )

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        value = value.view(batch_size * head_count, sub_seq_length, -1)
        attn_output = RingAV.apply(
            attn_weights,
            value.contiguous(),
            batch_size,
            self.num_heads,
            self.head_dim,
            sub_seq_length
        )

        return attn_output, attn_weights


if __name__ == '__main__':
    text = ['This is sample text']
    from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    inputs = tokenizer(text, return_tensors='pt')
    config = GPT2Config.from_pretrained('gpt2')
    model = GPT2WithSPModel(config)
    outputs = model(**inputs)
