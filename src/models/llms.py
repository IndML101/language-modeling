from src.base.modules import AbstractSeq2SeqModels
from src.modules.utils import Encoder, Decoder, PosEnc, GPTDecoder
from torch import nn, Tensor
import torch.nn.functional as F
import math
import torch


class Transformer(nn.Module):
    def __init__(
        self,
        device,
        n_tokens,
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
        super(Transformer, self).__init__()
        self.name = "Transformer"
        self.n_layers = n_layers
        self.d_model = d_model
        self.dff = dff
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        self.pos_encoding = PosEnc(device, self.d_model)
        # self.embedding = nn.Embedding(n_tokens, d_model)
        self.encoder = Encoder(
            self.pos_encoding,
            n_tokens=n_tokens,
            n_layers=n_layers,
            d_model=d_model,
            max_seq_length=max_seq_length,
            dff=dff,
            n_heads=n_heads,
            dropout=dropout,
            vdim=vdim,
            bias=bias,
            epsilon=epsilon,
        )
        self.decoder = Decoder(
            self.pos_encoding,
            n_tokens=n_tokens,
            n_layers=n_layers,
            d_model=d_model,
            max_seq_length=max_seq_length,
            dff=dff,
            n_heads=n_heads,
            dropout=dropout,
            vdim=vdim,
            bias=bias,
            epsilon=epsilon,
        )
        self.feed_forward = nn.Linear(d_model, n_tokens)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        for name, param in self.encoder.named_parameters():
            # nn.init.normal_(param.data)
            # print(name, param.requires_grad)
            nn.init.uniform_(param.data, -init_range, init_range)
        for name, param in self.decoder.named_parameters():
            # nn.init.normal_(param.data)
            nn.init.uniform_(param.data, -init_range, init_range)
        self.feed_forward.bias.data.zero_()
        # self.feed_forward.weight.data.normal_()
        self.feed_forward.weight.data.uniform_(-init_range, init_range)
        # self.embedding.bias.data.zero_()
        # self.embedding.weight.data.uniform_(-init_range, init_range)s

    def make_decoder_mask(self):
        pass

    def forward(self, src: Tensor, trg: Tensor, mask: Tensor):
        # src = self.embedding(src)
        # src = torch.div(src, math.sqrt(self.d_model))
        enc_out = self.encoder(src)
        # trg = self.embedding(trg)
        # trg = torch.div(trg, math.sqrt(self.d_model))
        enc_out = (
            enc_out[:, :, 0]
            .unsqueeze(-1)
            .expand(enc_out.shape[0], enc_out.shape[1], enc_out.shape[-1])
        )
        dec_out = self.decoder(trg, enc_out, mask)
        output = F.softmax(self.feed_forward(dec_out), dim=-1)
        # print(output)
        return output
        # return self.feed_forward(dec_out)

    def encode(self, src: Tensor) -> Tensor:
        src = src.reshape(-1, 1)
        # src = self.embedding(src)
        return self.encoder(torch.div(src, math.sqrt(self.d_model)))[0, 0, :].flatten()

    def decode(self, src: Tensor) -> Tensor:
        src = src.reshape(-1, 1)
        # src = self.embedding(src)
        enc_out = self.encoder(torch.div(src, math.sqrt(self.d_model)))
        dec_in = torch.zeros(src.shape)
        # dec_in = self.embedding(dec_in)
        return self.decoder(torch.div(dec_in, math.sqrt(self.d_model)), enc_out, None)[
            0, 0, :
        ].flatten()


class PreTrainedDecoder(nn.Module):
    def __init__(
        self,
        device,
        n_tokens,
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
        super(PreTrainedDecoder, self).__init__()
        self.name = "PreTrainedDecoder"
        self.n_layers = n_layers
        self.d_model = d_model
        self.dff = dff
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        self.pos_encoding = PosEnc(device, self.d_model)
        # self.embedding = nn.Embedding(n_tokens, d_model)
        self.decoder = Decoder(
            self.pos_encoding,
            n_tokens=n_tokens,
            n_layers=n_layers,
            d_model=d_model,
            max_seq_length=max_seq_length,
            dff=dff,
            n_heads=n_heads,
            dropout=dropout,
            vdim=vdim,
            bias=bias,
            epsilon=epsilon,
        )
        self.feed_forward = nn.Linear(d_model, n_tokens)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        for name, param in self.decoder.named_parameters():
            # nn.init.normal_(param.data)
            nn.init.uniform_(param.data, -init_range, init_range)
        self.feed_forward.bias.data.zero_()
        # self.feed_forward.weight.data.normal_()
        self.feed_forward.weight.data.uniform_(-init_range, init_range)
        # self.embedding.bias.data.zero_()
        # self.embedding.weight.data.uniform_(-init_range, init_range)s

    def make_decoder_mask(self):
        pass

    def forward(self, src: Tensor, mask: Tensor):
        # src = self.embedding(src)
        # src = torch.div(src, math.sqrt(self.d_model))
        # enc_out = self.encoder(src)
        # trg = self.embedding(trg)
        # trg = torch.div(trg, math.sqrt(self.d_model))
        # enc_out = (
        #     enc_out[:, :, 0]
        #     .unsqueeze(-1)
        #     .expand(enc_out.shape[0], enc_out.shape[1], enc_out.shape[-1])
        # )
        dec_out = self.decoder(src, mask=mask)
        output = F.softmax(self.feed_forward(dec_out), dim=-1)
        # print(output)
        return output
        # return self.feed_forward(dec_out)

    def encode(self, src: Tensor) -> Tensor:
        pass

    def decode(self, src: Tensor) -> Tensor:
        src = src.reshape(-1, 1)
        # src = self.embedding(src)
        # enc_out = self.encoder(torch.div(src, math.sqrt(self.d_model)))
        # dec_in = torch.zeros(src.shape)
        # dec_in = self.embedding(dec_in)
        return self.decoder(torch.div(src, math.sqrt(self.d_model)), src, None)[
            0, 0, :
        ].flatten()


class GPT(nn.Module):
    def __init__(
        self,
        device,
        n_tokens,
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
        super(GPT, self).__init__()
        self.name = "GPT"
        self.n_layers = n_layers
        self.d_model = d_model
        self.dff = dff
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        self.pos_encoding = PosEnc(device, self.d_model)
        # self.embedding = nn.Embedding(n_tokens, d_model)
        self.decoder = GPTDecoder(
            self.pos_encoding,
            n_tokens=n_tokens,
            n_layers=n_layers,
            d_model=d_model,
            max_seq_length=max_seq_length,
            dff=dff,
            n_heads=n_heads,
            dropout=dropout,
            vdim=vdim,
            bias=bias,
            epsilon=epsilon,
        )
        self.feed_forward = nn.Linear(d_model, n_tokens)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        for name, param in self.decoder.named_parameters():
            # nn.init.normal_(param.data)
            nn.init.uniform_(param.data, -init_range, init_range)
        self.feed_forward.bias.data.zero_()
        # self.feed_forward.weight.data.normal_()
        self.feed_forward.weight.data.uniform_(-init_range, init_range)
        # self.embedding.bias.data.zero_()
        # self.embedding.weight.data.uniform_(-init_range, init_range)s

    def make_decoder_mask(self):
        pass

    def forward(self, src: Tensor, mask: Tensor):
        # src = self.embedding(src)
        # src = torch.div(src, math.sqrt(self.d_model))
        # enc_out = self.encoder(src)
        # trg = self.embedding(trg)
        # trg = torch.div(trg, math.sqrt(self.d_model))
        # enc_out = (
        #     enc_out[:, :, 0]
        #     .unsqueeze(-1)
        #     .expand(enc_out.shape[0], enc_out.shape[1], enc_out.shape[-1])
        # )
        dec_out = self.decoder(src, mask=mask)
        output = F.softmax(self.feed_forward(dec_out), dim=-1)
        # print(output[-1,:,:].shape)
        return output[-1,:,:].unsqueeze(0)
        # return self.feed_forward(dec_out)

    def encode(self, src: Tensor) -> Tensor:
        pass

    def decode(self, src: Tensor) -> Tensor:
        src = src.reshape(-1, 1)
        # src = self.embedding(src)
        # enc_out = self.encoder(torch.div(src, math.sqrt(self.d_model)))
        # dec_in = torch.zeros(src.shape)
        # dec_in = self.embedding(dec_in)
        return self.decoder(torch.div(src, math.sqrt(self.d_model)), src, None)[
            0, :, :
        ].flatten()


class BERT(nn.Module, AbstractSeq2SeqModels):
    def __init__(self):
        pass
