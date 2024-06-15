from abc import ABC, abstractmethod
from typing import Optional, List
from torch import Tensor
from torch.utils.data import dataset
import torch
from torchtext.data.utils import get_tokenizer


class AbstractLoaders(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load(self):
        raise NotImplementedError


class AbstractVocabLoader(AbstractLoaders):
    def __init__(
        self,
        tokenizer: str = "basic_english",
        language: str = "en",
        specials: List[str] = ["<unk>", "<sep>", "<bos>", "<eos>"],
        min_freq: int = 2,
        special_first: bool = True,
        max_tokens: Optional[int] = None,
    ):
        super(AbstractVocabLoader, self).__init__()
        self.tokenizer = tokenizer
        self.language = language
        self.specials = specials
        # self.index_token = index_token
        self.min_freq = min_freq
        self.special_first = special_first
        self.max_tokens = max_tokens
        self.tokenizer_obj = get_tokenizer(self.tokenizer, self.language)



class AbstractDataProcessor(ABC):
    def __init__(self, text_loader: AbstractLoaders, vocab_loader: AbstractVocabLoader):
        self.train_iter, self.val_iter, self.test_iter = text_loader.load()
        self.vocab_loader = vocab_loader
        self.vocab = self.vocab_loader.load()

    def process_data(self, text_iter: dataset.IterableDataset) -> Tensor:
        data = [
            torch.tensor(self.vocab(self.vocab_loader.tokenizer_obj(item)), dtype=torch.long)
            for item in text_iter
        ]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def run(self):
        # train_iter, val_iter, test_iter = self.text_loader.load()
        train_data = self.process_data(self.train_iter)
        val_data = self.process_data(self.val_iter)
        test_data = self.process_data(self.test_iter)

        return train_data, val_data, test_data

    def get_vocab_size(self):
        return len(self.vocab)


class AbstractBatchLoader(AbstractLoaders):
    def __init__(self, train_data, val_data, test_data, batch_size=32):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

    def make_batch(self, data: Tensor, batch_size: int):
        seq_len = data.size(0) // batch_size
        data = data[: seq_len * batch_size]
        data = data.view(seq_len, batch_size).contiguous()
        return data

    def run(self):
        train = self.make_batch(self.train_data, self.batch_size)
        val = self.make_batch(self.val_data, self.batch_size)
        test = self.make_batch(self.test_data, self.batch_size)
        return train, val, test


class AbstractTrainers(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make_target(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, filename):
        raise NotImplementedError

    @abstractmethod
    def get_model_architecture(self):
        raise NotImplementedError

    @abstractmethod
    def _train(self):
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError


class AbstractSeq2SeqModels(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def encode(self):
        raise NotImplementedError

    @abstractmethod
    def decode(self):
        raise NotImplementedError


class AbstractExperiment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        raise NotImplementedError
