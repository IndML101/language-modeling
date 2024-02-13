from src.base import AbstractLoaders, AbstractVocabLoader
from torchtext.datasets import WikiText2, PennTreebank
from torchtext.vocab import build_vocab_from_iterator


class WikiText2Loader(AbstractLoaders):
    def load(self):
        train_iter, val_iter, test_iter = WikiText2()
        return train_iter, val_iter, test_iter


class WikiText2VocabLoader(AbstractVocabLoader):
    def __init__(self):
        super(WikiText2VocabLoader, self).__init__()

    def load(self):
        train_iter = WikiText2(split="train")
        vocab = build_vocab_from_iterator(
            map(self.tokenizer_obj, train_iter),
            min_freq=self.min_freq,
            specials=self.specials,
            special_first=self.special_first,
            max_tokens=self.max_tokens,
        )
        # vocab.set_default_index(vocab[self.index_token])
        return vocab


class PTBLoader(AbstractLoaders):
    def load(self):
        train_iter, val_iter, test_iter = PennTreebank()
        return train_iter, val_iter, test_iter


class PTBVocabLoader(AbstractVocabLoader):
    def __init__(self):
        super(PTBVocabLoader, self).__init__()
        

    def load(self):
        train_iter = PennTreebank(split="train")
        vocab = build_vocab_from_iterator(
            map(self.tokenizer_obj, train_iter),
            min_freq=self.min_freq,
            specials=self.specials,
            special_first=self.special_first,
            max_tokens=self.max_tokens,
        )
        vocab.set_default_index(vocab["<unk>"])
        return vocab
