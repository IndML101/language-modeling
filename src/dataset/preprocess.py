# import torch
# from torch import Tensor
# from torch.utils.data import dataset
from src.base import AbstractDataProcessor
from src.dataset.loaders import (
    WikiText2Loader,
    WikiText2VocabLoader,
    PTBLoader,
    PTBVocabLoader,
    WikiText103Loader,
    WikiText103VocabLoader
)


class WikiTextDataProcessor(AbstractDataProcessor):
    def __init__(
        self,
        text_loader: WikiText2Loader = WikiText2Loader(),
        vocab_loader: WikiText2VocabLoader = WikiText2VocabLoader(),
    ):
        super(WikiTextDataProcessor, self).__init__(text_loader, vocab_loader)


class PTBDataProcessor(AbstractDataProcessor):
    def __init__(
        self,
        text_loader: PTBLoader = PTBLoader(),
        vocab_loader: PTBVocabLoader = PTBVocabLoader(),
    ):
        super(PTBDataProcessor, self).__init__(text_loader, vocab_loader)


class WikiText103DataProcessor(AbstractDataProcessor):
    def __init__(
        self,
        text_loader: WikiText103Loader = WikiText103Loader(),
        vocab_loader: WikiText103VocabLoader = WikiText103VocabLoader(),
    ):
        super(WikiText103DataProcessor, self).__init__(text_loader, vocab_loader)