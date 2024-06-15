import os
import random
import numpy as np
from src.base.modules import AbstractTrainers, AbstractDataProcessor
from src.models.llms import Transformer, PreTrainedDecoder, GPT
from src.dataset.batches import EncoderDecoderLLMBatchLoader, DecoderLLMBatchLoader
from src.dataset.preprocess import WikiTextDataProcessor
from src.train.utils import LRPolicyAdam
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR
import torch
import pickle as pkl
import math
import time
from datetime import datetime


class TransformerTrainer(AbstractTrainers):
    def __init__(
        self,
        model: str = None,
        text_processor: AbstractDataProcessor or str = WikiTextDataProcessor(),
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        decay_steps: float = 1.0,
        **model_kwargs,
    ):
        super(TransformerTrainer, self).__init__()
        self.__dict__.update(**model_kwargs)
        self.kwargs = model_kwargs
        self.device = self.kwargs['device']
        self.batch_size = batch_size
        if isinstance(text_processor, str):
            self.text_processor = self.load_text_processor(text_processor)
        else:
            self.text_processor = text_processor
        train, val, test = self.text_processor.run()
        self.n_tokens = self.text_processor.get_vocab_size()
        self.model = Transformer(
            n_tokens=self.n_tokens, **model_kwargs
        )
        if model is not None:
            self.model.load_state_dict(torch.load(model))
        self.max_seq_length = self.model.max_seq_length
        self.model.to(self.device)
        # self.lr = self.get_learning_rate(1)
        self.lr = learning_rate
        self.optimizer = Adam(
            self.model.parameters(), self.lr, betas=(0.9, 0.98), eps=1e-9
        )
        # self.optimizer = SGD(self.model.parameters(), 1e-6)
        self.enc_batch_loader = EncoderDecoderLLMBatchLoader(
            train, val, test, batch_size
        )
        self.train, self.val, self.test = self.enc_batch_loader.run()
        # self.lr_scheduler = LambdaLR(
        #     self.optimizer, lr_lambda=LRPolicyAdam(self.model.d_model)
        # )
        self.lr_scheduler = StepLR(self.optimizer, decay_steps, gamma=0.95)
        self.criterion = torch.nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=0.1
        )

    def get_learning_rate(self, step_num):
        warmup_steps = 10
        return (
            1
            / math.sqrt(self.model.d_model)
            * min(1 / math.sqrt(step_num), step_num / math.pow(warmup_steps, 3 / 2))
        )

    def get_model_architecture(self):
        print(self.model)

    def save(self, filename: str):
        torch.save(self.model.state_dict(), filename + ".pt")
        with open(filename + ".dt", "wb") as f:
            pkl.dump(self.text_processor, f)

    def load_text_processor(self, filename):
        with open(filename, "rb") as f:
            text_processor = pkl.load(f)
        return text_processor

    def make_target(self, trg: torch.Tensor) -> torch.Tensor:
        traget = torch.zeros((trg.shape[0], self.batch_size, self.n_tokens))
        for s in range(trg.shape[0]):
            for b in range(self.batch_size):
                traget[s, b, trg[s, b]] = 1

        return traget

    def make_target_mask(self, trg: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size = trg.size()
        trg_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=1).expand(
            batch_size, seq_len, seq_len
        )
        # print(trg_mask)
        return trg_mask

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _train(self, epoch: int):
        self.model.train()
        total_loss = 0
        log_interval = 100
        start_time = time.time()

        num_batches = self.train.shape[0] // self.max_seq_length
        for batch, i in enumerate(range(num_batches)):
        # for batch, i in enumerate(range(0, self.train.size(0) - 1, self.batch_size)):
            idx = np.random.randint(0, self.train.size(0)-self.max_seq_length-1)
            # shuffle_idx = np.random.shuffle(np.arange(len(self.train))).to_list()
            # input_data = self.train[shuffle_idx,:]
            # random.shuffle(self.train)
            enc_src, enc_trg = self.enc_batch_loader.load(
                self.train, idx, self.max_seq_length
            )
            dec_src, dec_trg = self.enc_batch_loader.load_decoder_batch(
                self.train, idx + enc_src.shape[0], enc_src.shape[0]
            )
            dec_trg_mask = self.make_target_mask(dec_trg)
            output = self.model(
                enc_src.to(self.device),
                dec_src.to(self.device),
                dec_trg_mask.to(self.device),
            )
            target = self.make_target(dec_trg)
            loss = self.criterion(
                output.transpose(0, 1).transpose(1, -1),
                target.to(self.device).transpose(0, 1).transpose(1, -1),
            )
            # loss2 = torch.nn.functional.cross_entropy(output, target.to(self.device))

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            step_loss = loss.item()
            # print(step_loss)
            total_loss += step_loss

            if batch % log_interval == 0 and batch > 0:
                lr = self.lr_scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(
                    f"| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | "
                    f"lr {lr:02.2e} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}"
                )
                total_loss = 0
                start_time = time.time()

    def _evaluate(self):
        self.model.eval()  # turn on evaluation mode
        total_loss = 0.0
        num_batches = self.val.size(0)//self.max_seq_length
        with torch.no_grad():
            for i in range(0, self.val.size(0) - 1, self.max_seq_length):
                enc_src, enc_trg = self.enc_batch_loader.load(
                    self.val, i, self.max_seq_length
                )
                dec_src, dec_trg = self.enc_batch_loader.load_decoder_batch(
                    self.val, i + enc_src.shape[0], enc_src.shape[0]
                )
                dec_trg_mask = self.make_target_mask(dec_trg)
                seq_len = enc_src.size(0)
                # print(enc_src.shape)
                output = self.model(
                    enc_src.to(self.device),
                    dec_src.to(self.device),
                    dec_trg_mask.to(device=self.device),
                )
                # output_flat = output.view(-1, ntokens)
                target = self.make_target(dec_trg)
                total_loss += self.criterion(
                        output.transpose(0, 1).transpose(1, -1),
                        target.to(self.device).transpose(0, 1).transpose(1, -1),
                    ).item()

        return total_loss / num_batches

    def run(self, n_epochs: int = 10000, last_val_loss: float = float("inf")):
        best_val_loss = last_val_loss
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        filename = (
            self.model.name
            + "-n{}-d{}-dff{}-h{}-".format(
                self.model.n_layers,
                self.model.d_model,
                self.model.dff,
                self.model.n_heads,
            )
            + now_str
        )

        print(
            "Training {} with {} parameters".format(
                self.model.name, self.count_parameters()
            )
        )

        for epoch in range(1, n_epochs + 1):
            epoch_start_time = time.time()
            self._train(epoch)
            val_loss = self._evaluate()
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print("-" * 95)
            print(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}"
            )
            print("-" * 95)

            if val_loss < best_val_loss:
                # now = datetime.now()
                # now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
                # filename = self.model.name + "_" + now_str

                best_val_loss = val_loss
                self.save(os.path.join("Artifacts", filename))

            self.lr_scheduler.step()
        # model.load_state_dict(torch.load(best_model_params_path))

    def generate_text(self, enc_text, dec_text, n_tokes=100):
        self.model.eval()
        with torch.no_grad():
            enc_src = self.text_processor.process_data(enc_text).unsqueeze(1).to(self.device, dtype=torch.long)
            dec_src = self.text_processor.process_data(dec_text).unsqueeze(1).to(self.device, dtype=torch.long)
            tokens = list()

            # print(dec_src.shape)

            for i in range(n_tokes):
                # print(i)
                output = self.model(
                    enc_src,
                    dec_src,
                    None
                )

                # print(torch.argmax(dec_src, -1).squeeze(-1))
                output = torch.argmax(output, -1)
                # print(output.shape)
                tokens.append(output.squeeze(-1)[-1].item())
                dec_src = torch.concat((dec_src, output[-1].unsqueeze(1)), 0)

            return ' '.join(self.text_processor.vocab.lookup_tokens(tokens))



class TransformerDecoderTrainer(TransformerTrainer):
    def __init__(
        self,
        model: str = None,
        text_processor: AbstractDataProcessor or str = WikiTextDataProcessor(),
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        decay_steps: float = 1.0,
        **model_kwargs,
    ):
        super(TransformerDecoderTrainer, self).__init__(
            model=model,
            text_processor=text_processor,
            batch_size=batch_size,
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            **model_kwargs,
        )
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.batch_size = batch_size
        if isinstance(text_processor, str):
            self.text_processor = self.load_text_processor(text_processor)
        else:
            self.text_processor = text_processor
        train, val, test = self.text_processor.run()
        # self.n_tokens = self.text_processor.get_vocab_size()
        self.model = PreTrainedDecoder(
            device=self.device, n_tokens=self.n_tokens, **model_kwargs
        )
        if model is not None:
            self.model.load_state_dict(torch.load(model))
        # self.max_seq_length = self.model.max_seq_length
        self.model.to(self.device)
        # self.lr = self.get_learning_rate(1)
        # self.lr = learning_rate
        self.optimizer = Adam(
            self.model.parameters(), self.lr, betas=(0.9, 0.98), eps=1e-9
        )
        # self.optimizer = SGD(self.model.parameters(), 1e-6)
        self.dec_batch_loader = DecoderLLMBatchLoader(train, val, test, batch_size)
        self.train, self.val, self.test = self.dec_batch_loader.run()
        # self.lr_scheduler = LambdaLR(
        #     self.optimizer, lr_lambda=LRPolicyAdam(self.model.d_model)
        # )
        self.lr_scheduler = StepLR(self.optimizer, decay_steps, gamma=0.95)
        self.criterion = torch.nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=0.1
        )

    def _train(self, epoch: int):
        self.model.train()
        total_loss = 0
        log_interval = 100
        start_time = time.time()

        num_batches = self.train.shape[0] // self.batch_size
        for batch, i in enumerate(range(0, self.train.size(0) - 1, self.batch_size)):
            src, trg = self.dec_batch_loader.load(self.train, i, self.max_seq_length)
            # dec_src, dec_trg = self.enc_batch_loader.load_decoder_batch(
            #     self.train, i + enc_src.shape[0], enc_src.shape[0]
            # )
            trg_mask = self.make_target_mask(trg)
            output = self.model(
                src.to(self.device),
                trg_mask.to(self.device),
            )
            target = self.make_target(trg)
            loss = self.criterion(
                output.transpose(0, 1).transpose(1, -1),
                target.to(self.device).transpose(0, 1).transpose(1, -1),
            )
            # loss2 = torch.nn.functional.cross_entropy(output, target.to(self.device))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            step_loss = loss.item()
            total_loss += step_loss

            if batch % log_interval == 0 and batch > 0:
                lr = self.lr_scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(
                    f"| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | "
                    f"lr {lr:02.2e} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}"
                )
                total_loss = 0
                start_time = time.time()

    def _evaluate(self):
        self.model.eval()  # turn on evaluation mode
        total_loss = 0.0
        with torch.no_grad():
            for i in range(0, self.val.size(0) - 1, self.batch_size):
                src, trg = self.dec_batch_loader.load(self.val, i, self.max_seq_length)
                # dec_src, dec_trg = self.enc_batch_loader.load_decoder_batch(
                #     self.val, i + enc_src.shape[0], enc_src.shape[0]
                # )
                trg_mask = self.make_target_mask(trg)
                seq_len = src.size(0)
                output = self.model(
                    src.to(self.device),
                    trg_mask.to(device=self.device),
                )
                # output_flat = output.view(-1, ntokens)
                target = self.make_target(trg)
                total_loss += (
                    seq_len
                    * self.criterion(
                        output.transpose(0, 1).transpose(1, -1),
                        target.to(self.device).transpose(0, 1).transpose(1, -1),
                    ).item()
                )
        return total_loss / (self.val.size(0) - 1)


class GPTTrainer(TransformerDecoderTrainer):
    def __init__(
        self,
        model: str = None,
        text_processor: AbstractDataProcessor or str = WikiTextDataProcessor(),
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        decay_steps: float = 1.0,
        **model_kwargs,
    ):
        super(GPTTrainer, self).__init__(
            model=model,
            text_processor=text_processor,
            batch_size=batch_size,
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            **model_kwargs,
        )
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.batch_size = batch_size
        if isinstance(text_processor, str):
            self.text_processor = self.load_text_processor(text_processor)
        else:
            self.text_processor = text_processor
        train, val, test = self.text_processor.run()
        # self.n_tokens = self.text_processor.get_vocab_size()
        self.model = GPT(device=self.device, n_tokens=self.n_tokens, **model_kwargs)
        if model is not None:
            self.model.load_state_dict(torch.load(model))
        # self.max_seq_length = self.model.max_seq_length
        self.model.to(self.device)
        # self.lr = self.get_learning_rate(1)
        # self.lr = learning_rate
        self.optimizer = Adam(
            self.model.parameters(), self.lr, betas=(0.9, 0.98), eps=1e-9
        )
        # self.optimizer = SGD(self.model.parameters(), 1e-6)
        self.dec_batch_loader = DecoderLLMBatchLoader(train, val, test, batch_size)
        self.train, self.val, self.test = self.dec_batch_loader.run()
        # self.lr_scheduler = LambdaLR(
        #     self.optimizer, lr_lambda=LRPolicyAdam(self.model.d_model)
        # )
        self.lr_scheduler = StepLR(self.optimizer, decay_steps, gamma=0.95)
        self.criterion = torch.nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=0.1
        )

    def make_target(self, trg: torch.Tensor) -> torch.Tensor:
        traget = torch.zeros((1, self.batch_size, self.n_tokens))
        for b in range(self.batch_size):
            traget[0, b, trg[-1, b]] = 1

        return traget
