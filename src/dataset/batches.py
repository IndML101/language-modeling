from typing import Tuple
from torch import Tensor
from src.base.modules import AbstractBatchLoader


class EncoderDecoderLLMBatchLoader(AbstractBatchLoader):
    def __init__(self, train_data, val_data, test_data, batch_size=32):
        super(EncoderDecoderLLMBatchLoader, self).__init__(
            train_data, val_data, test_data, batch_size
        )

    def load(self, data: Tensor, idx: int, max_seq_len: int) -> Tuple[Tensor, Tensor]:
        seq_len = min(max_seq_len, len(data) - 1 - idx)
        source = data[idx : idx + seq_len // 2, :]
        target = data[idx + seq_len // 2 : idx + seq_len, :]
        return source, target

    def load_decoder_batch(self, data: Tensor, idx: int, max_seq_len: int) -> Tuple[Tensor, Tensor]:
        seq_len = min(max_seq_len, len(data) - 1 - idx)
        source = data[idx : idx + seq_len, :]
        target = data[idx + 1 : idx + 1 + seq_len, :]
        return source, target



class DecoderLLMBatchLoader(AbstractBatchLoader):
    def __init__(self, train_data, val_data, test_data, batch_size=32):
        super(DecoderLLMBatchLoader, self).__init__(
            train_data, val_data, test_data, batch_size
        )

    def load(self, data: Tensor, idx: int, max_seq_len: int) -> Tuple[Tensor, Tensor]:
        seq_len = min(max_seq_len, len(data) - 1 - idx)
        source = data[idx : idx + seq_len, :]
        target = data[idx + 1 : idx + 1 + seq_len, :]
        return source, target


class BERTBatchLoader(AbstractBatchLoader):
    def __init__(self):
        super(BERTBatchLoader, self).__init__()

    def make_mask(self, data: Tensor, n_mask: int) -> Tensor:
        # input mask
        # output mask
        pass

    def load(data: Tensor, idx: int, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # get sequence
        # masked input sequence
        # masked output sequence
        pass
