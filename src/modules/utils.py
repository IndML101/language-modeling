from torch import nn, Tensor
import torch.nn.functional as F
import torch
from typing import Optional
import math


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_seq_length: int = 30):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term_even = 10 ** (4 * 2 * torch.arange(0, d_model, 2) / d_model)
        div_term_odd = torch.pow(
            (10 * torch.ones(d_model // 2)),
            (4 * 2 * torch.arange(1, d_model, 2) / d_model),
        )
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position / div_term_even)
        pe[:, 0, 1::2] = torch.cos(position / div_term_odd)
        self.register_buffer("pe", pe)

    def forward(self, src: Tensor) -> Tensor:
        # src = src + torch.autograd.Variable(self.pe[:src.size(0)], requires_grad=False)
        src = src + self.pe[: src.size(0), :, :]
        # return self.dropout(src)
        return src


class PosEnc:
    def __init__(self, device, d_model: int = 128):
        super(PosEnc, self).__init__()
        self.d_model = d_model
        self.device = device

    def get_pe(self, seq_length: int) -> Tensor:
        position = torch.arange(seq_length).unsqueeze(1)
        div_term_even = 10 ** (4 * 2 * torch.arange(0, self.d_model, 2) / self.d_model)
        div_term_odd = torch.pow(
            (10 * torch.ones(self.d_model // 2)),
            (4 * 2 * torch.arange(1, self.d_model, 2) / self.d_model),
        )
        pe = torch.zeros(seq_length, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position / div_term_even)
        pe[:, 0, 1::2] = torch.cos(position / div_term_odd)

        return pe.to(self.device)


class PositionalEmbedding(nn.Module):
    def __init__(self, n_tokens: int, d_model: int = 128, max_seq_length: int = 30):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term_even = 10 ** (4 * 2 * torch.arange(0, d_model, 2) / d_model)
        div_term_odd = torch.pow(
            (10 * torch.ones(d_model // 2)),
            (4 * 2 * torch.arange(1, d_model, 2) / d_model),
        )
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position / div_term_even)
        pe[:, 0, 1::2] = torch.cos(position / div_term_odd)
        self.register_buffer("pe", pe)

    def forward(self, src: Tensor) -> Tensor:
        src = self.embedding(src)
        src = src + self.pe[: src.size(0), :, :]
        # return self.dropout(src)
        return src


class AttentionHead(nn.Module):
    def __init__(
        self, kdim: int, vdim: int = 32, d_model: int = 128, bias: bool = True
    ) -> Tensor:
        super(AttentionHead, self).__init__()
        self.kdim = kdim
        self.wk = nn.Linear(d_model, kdim, bias=bias)
        self.wq = nn.Linear(d_model, kdim, bias=bias)
        self.wv = nn.Linear(d_model, vdim, bias=bias)

    def forward(
        self, key: Tensor, query: Tensor, value: Tensor, mask: Tensor
    ) -> Tensor:
        k = self.wk(key)
        q = self.wq(query)
        v = self.wv(value)
        prod = torch.matmul(q.transpose(0, 1), k.transpose(0, 1).transpose(-1, -2))

        if mask is not None:
            prod = prod.masked_fill(mask == 0, float("-inf"))

        prod = torch.div(prod, math.sqrt(self.kdim))
        alpha = F.softmax(prod, dim=-1)
        return torch.matmul(alpha, v.transpose(0, 1)).transpose(0, 1)


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        vdim: int = 32,
        bias: bool = True,
    ) -> Tensor:
        super(MultiHeadedAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.kdim = self.d_model // self.n_heads
        self.attn_heads = nn.ModuleList(
            [
                AttentionHead(
                    kdim=self.kdim, vdim=vdim, d_model=self.d_model, bias=bias
                )
                for i in range(self.n_heads)
            ]
        )
        self.wo = nn.Sequential(
            nn.Linear(self.n_heads * vdim, self.d_model, bias=bias),
            nn.ReLU()
        )

    def forward(
        self, key: Tensor, query: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        multi_head_output = list()
        for head in self.attn_heads:
            multi_head_output.append(head(key, query, value, mask))

        return self.wo(torch.cat(multi_head_output, dim=-1))


class FFNLayer(nn.Module):
    def __init__(self, d_model: int = 128, dff: int = 512, bias: bool = True):
        super(FFNLayer, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff, bias=bias),
            nn.ReLU(),
            nn.Linear(dff, d_model, bias=bias),
        )

    def forward(self, src: Tensor) -> Tensor:
        return self.ffn(src)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        dff: int = 512,
        n_heads: int = 4,
        dropout: float = 0.1,
        vdim: int = 32,
        bias: bool = True,
        epsilon: float = 1e-5,
    ):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            d_model=d_model, n_heads=n_heads, vdim=vdim, bias=bias
        )
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=epsilon, bias=bias)
        self.feed_forward = FFNLayer(d_model=d_model, dff=dff, bias=bias)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=epsilon, bias=bias)

    def forward(self, src: Tensor) -> Tensor:
        trg = self.layer_norm1(self.dropout1(src + self.self_attn(src, src, src)))
        trg = self.layer_norm2(self.dropout2(trg + self.feed_forward(trg)))
        return trg


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        dff: int = 512,
        n_heads: int = 4,
        dropout: float = 0.1,
        vdim: int = 32,
        bias: bool = True,
        epsilon: float = 1e-5,
    ):
        super(DecoderLayer, self).__init__()
        self.masked_multihead_attn = MultiHeadedAttention(
            d_model=d_model, n_heads=n_heads, vdim=vdim, bias=bias
        )
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=epsilon, bias=bias)
        self.multihead_attn = MultiHeadedAttention(
            d_model=d_model, n_heads=n_heads, vdim=vdim, bias=bias
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=epsilon, bias=bias)
        self.feed_forward = FFNLayer(d_model=d_model, dff=dff, bias=bias)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm3 = nn.LayerNorm(d_model, eps=epsilon, bias=bias)

    def forward(self, src: Tensor, memory: Tensor, mask: Tensor) -> Tensor:
        trg = self.layer_norm1(
            src + self.dropout1(self.masked_multihead_attn(src, src, src, mask))
        )
        trg = self.layer_norm2(
            trg + self.dropout2(self.multihead_attn(memory, trg, memory))
        )
        # trg = self.layer_norm2(
        #     trg + self.dropout2(self.multihead_attn(memory, memory, trg, mask))
        # )
        trg = self.layer_norm3(trg + self.dropout3(self.feed_forward(trg)))
        return trg


class GPTDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        dff: int = 512,
        n_heads: int = 4,
        dropout: float = 0.1,
        vdim: int = 32,
        bias: bool = True,
        epsilon: float = 1e-5,
    ):
        super(GPTDecoderLayer, self).__init__()
        self.masked_multihead_attn = MultiHeadedAttention(
            d_model=d_model, n_heads=n_heads, vdim=vdim, bias=bias
        )
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=epsilon, bias=bias)
        # self.multihead_attn = MultiHeadedAttention(
        #     d_model=d_model, n_heads=n_heads, vdim=vdim, bias=bias
        # )
        # self.dropout2 = nn.Dropout(dropout)
        # self.layer_norm2 = nn.LayerNorm(d_model, eps=epsilon, bias=bias)
        self.feed_forward = FFNLayer(d_model=d_model, dff=dff, bias=bias)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm3 = nn.LayerNorm(d_model, eps=epsilon, bias=bias)

    def forward(self, src: Tensor, memory: Tensor, mask: Tensor) -> Tensor:
        trg = self.layer_norm1(
            src + self.dropout1(self.masked_multihead_attn(src, src, src, mask))
        )
        # trg = self.layer_norm2(
        #     trg + self.dropout2(self.multihead_attn(memory, memory, trg))
        # )
        # trg = self.layer_norm2(
        #     trg + self.dropout2(self.multihead_attn(memory, memory, trg, mask))
        # )
        trg = self.layer_norm3(trg + self.dropout3(self.feed_forward(trg)))
        return trg


class Encoder(nn.Module):
    def __init__(
        self,
        pos_encoding: PosEnc,
        n_tokens: int,
        n_layers: int = 3,
        d_model: int = 128,
        max_seq_length: int = 30,
        dff: int = 512,
        n_heads: int = 4,
        dropout: float = 0.1,
        vdim: int = 32,
        bias: bool = True,
        epsilon: float = 1e-5,
    ):
        super(Encoder, self).__init__()
        # self.pe = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_length)
        self.d_model = d_model
        # self.pe = PositionalEmbedding(n_tokens, d_model=d_model, max_seq_length=max_seq_length)
        self.pos_encoding = pos_encoding
        self.pe = nn.Embedding(n_tokens, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    dff=dff,
                    n_heads=n_heads,
                    dropout=dropout,
                    vdim=vdim,
                    bias=bias,
                    epsilon=epsilon,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, src: Tensor) -> Tensor:
        pe_embedding = self.pe(src)
        pe_embedding = torch.div(pe_embedding, math.sqrt(self.d_model))
        pe_embedding = pe_embedding + self.pos_encoding.get_pe(src.shape[0])
        trg = self.dropout(pe_embedding)
        for layer in self.encoder_layers:
            trg = layer(trg)

        return trg


class Decoder(nn.Module):
    def __init__(
        self,
        pos_encoding: PosEnc,
        n_tokens: int,
        n_layers: int = 3,
        d_model: int = 128,
        max_seq_length: int = 30,
        dff: int = 512,
        n_heads: int = 4,
        dropout: float = 0.1,
        vdim: int = 32,
        bias: bool = True,
        epsilon: float = 1e-5,
    ):
        super(Decoder, self).__init__()
        # self.pe = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_length)
        self.d_model = d_model
        # self.pe = PositionalEmbedding(n_tokens, d_model=d_model, max_seq_length=max_seq_length)
        self.pos_encoding = pos_encoding
        self.pe = nn.Embedding(n_tokens, d_model)
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    dff=dff,
                    n_heads=n_heads,
                    dropout=dropout,
                    vdim=vdim,
                    bias=bias,
                    epsilon=epsilon,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, src: Tensor, memory: Tensor = None, mask: Tensor = None) -> Tensor:
        pe_embedding = self.pe(src)
        pe_embedding = torch.div(pe_embedding, math.sqrt(self.d_model))
        pe_embedding = pe_embedding + self.pos_encoding.get_pe(src.shape[0])
        trg = self.dropout(pe_embedding)
        if memory is not None:
            for layer in self.decoder_layers:
                trg = layer(trg, memory, mask)
        else:
            for layer in self.decoder_layers:
                trg = layer(trg, trg, mask)
        return trg


class GPTDecoder(nn.Module):
    def __init__(
        self,
        pos_encoding: PosEnc,
        n_tokens: int,
        n_layers: int = 3,
        d_model: int = 128,
        max_seq_length: int = 30,
        dff: int = 512,
        n_heads: int = 4,
        dropout: float = 0.1,
        vdim: int = 32,
        bias: bool = True,
        epsilon: float = 1e-5,
    ):
        super(GPTDecoder, self).__init__()
        # self.pe = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_length)
        self.d_model = d_model
        # self.pe = PositionalEmbedding(n_tokens, d_model=d_model, max_seq_length=max_seq_length)
        self.pos_encoding = pos_encoding
        self.pe = nn.Embedding(n_tokens, d_model)
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList(
            [
                GPTDecoderLayer(
                    d_model=d_model,
                    dff=dff,
                    n_heads=n_heads,
                    dropout=dropout,
                    vdim=vdim,
                    bias=bias,
                    epsilon=epsilon,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, src: Tensor, mask: Tensor = None) -> Tensor:
        pe_embedding = self.pe(src)
        pe_embedding = torch.div(pe_embedding, math.sqrt(self.d_model))
        pe_embedding = pe_embedding + self.pos_encoding.get_pe(src.shape[0])
        trg = self.dropout(pe_embedding)
        for layer in self.decoder_layers:
            trg = layer(trg, trg, mask)

        return trg