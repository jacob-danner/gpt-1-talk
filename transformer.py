from torch import nn
import torch
from einops import rearrange

d_model = 768
vocab_size = 40478
context_size = 512
n_layers = 12
n_attention_heads = 12
d_attention_head = d_model // n_attention_heads


class Attention(nn.Module):
    def __init__(self):
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, residual_stream):  # 'batch tokens d_model'
        # project from residual stream
        q = self.W_q(residual_stream)  # 'batch tokens d_model'
        k = self.W_k(residual_stream)  # 'batch tokens d_model'
        v = self.W_v(residual_stream)  # 'batch tokens d_model'

        # split into n_heads
        q = rearrange(
            q,
            'batch tokens (n_heads d_head) -> batch n_heads tokens d_head',
            n_heads=n_attention_heads,
            d_head=d_attention_head,
        )
        k = rearrange(
            k,
            'batch tokens (n_heads d_head) -> batch n_heads tokens d_head',
            n_heads=n_attention_heads,
            d_head=d_attention_head,
        )
        v = rearrange(
            v,
            'batch tokens (n_heads d_head) -> batch n_heads tokens d_head',
            n_heads=n_attention_heads,
            d_head=d_attention_head,
        )

        # q @ k_t: 'batch n_heads tokens d_head' @ 'batch n_heads d_head tokens'
        k_t = rearrange(k, 'batch n_heads tokens d_head -> batch n_heads d_head tokens')
        attention_scores = q @ k_t  # 'batch n_heads tokens tokens'

        # causal mask
        attention_scores = torch.tril(
            attention_scores
        )  # zero out tokens that "haven't been revealed yet"

        attention_scores = self.softmax(attention_scores / torch.sqrt(d_attention_head))
        weighted_averages = attention_scores @ v  # 'batch n_heads tokens d_head'

        # concatenate head outputs and project (stir each attetnion head's insights into one)
        weighted_averages = rearrange(
            weighted_averages, 'batch n_heads tokens d_head -> batch tokens (n_heads d_head)'
        )
        return self.W_o(weighted_averages)


class TransformerBlock(nn.Module):
    def __init__(self):
        self.attention = Attention()
        self.MLP = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, residual_stream):
        attention_output = self.attention(residual_stream)
        # layer norm here in gpt 1
        residual_stream = torch.add(residual_stream, attention_output)

        mlp_output = self.MLP(residual_stream)
        # layer norm here in gpt 1
        residual_stream = torch.add(residual_stream, mlp_output)

        return residual_stream


class Transformer(nn.Module):
    def __init__(self):
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_size, d_model)
        self.h = nn.Sequential(*[TransformerBlock() for _ in range(n_layers)])
        self.unembed = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tokens):  # 'batch tokens'
        residual_stream = self.token_embed(tokens) + self.pos_embed(
            tokens
        )  # note that "self.pos_embed(tokens)" is not valid, we would want position ids
        residual_stream = self.h(residual_stream)
        distribution = self.softmax(self.unembed(residual_stream))
        return distribution
