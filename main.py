from src.train.trainers import TransformerTrainer, TransformerDecoderTrainer, GPTTrainer
from src.dataset.preprocess import (
    WikiTextDataProcessor,
    PTBDataProcessor,
    WikiText103DataProcessor,
)
import math
import torch
import numpy as np


class ModelConfig:
    def __init__(
        self,
        text_processor,
        learning_rate,
        batch_size,
        decay_steps,
        model_name=None,
        tstamp=None,
        **kwargs,
    ):
        self.__dict__.update(**kwargs)
        self.kwargs = kwargs
        self.text_processor = text_processor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.decay_steps = decay_steps
        if model_name is not None and tstamp is not None:
            self.model_name = "Artifacts/{}-n{}-d{}-dff{}-h{}-{}.pt".format(
                model_name,
                self.kwargs["n_layers"],
                self.kwargs["d_model"],
                self.kwargs["dff"],
                self.kwargs["n_heads"],
                tstamp,
            )
        else:
            self.model_name = None
        

    def trainllm(self, model, n_epochs):
        last_val_loss = model._evaluate()
        print('-'*89)
        print('*'*40, round(math.exp(last_val_loss),2), '*'*40)
        print('-'*89)
        model.run(n_epochs, last_val_loss)



if __name__ == "__main__":

    CUDA_LAUNCH_BLOCKING=1

    # model arguments
    tstamp = '2024-06-15-15-15-47'
    model_name = 'Transformer'
    # text_processor = PTBDataProcessor()
    text_processor = WikiTextDataProcessor()
    # text_processor = WikiText103DataProcessor()
    learning_rate = 5e-7
    batch_size = 128
    decay_steps = 5
    n_epochs = 20

    trans_args = dict(
        n_layers=6,
        d_model=256,
        max_seq_length=35,
        dff=1024,
        n_heads=8,
        dropout=0.1,
        vdim=32,
        bias=True,
        epsilon=1e-5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # model config
    transformer_config = ModelConfig(
        text_processor=text_processor,
        learning_rate=learning_rate,
        batch_size=batch_size,
        decay_steps=decay_steps,
        model_name=model_name,
        tstamp=tstamp,
        **trans_args,
    )

    # transformer
    transformer = TransformerTrainer(
        model=transformer_config.model_name,
        text_processor=transformer_config.text_processor,
        learning_rate=transformer_config.learning_rate,
        batch_size=transformer_config.batch_size,
        decay_steps=transformer_config.decay_steps,
        **transformer_config.kwargs,
    )

    transformer_config.trainllm(transformer, n_epochs)
    enc_text = ['What', 'a', 'day', '.']
    dec_text = ['I', 'believe', 'it']
    output = transformer.generate_text(enc_text, dec_text, n_tokes=200)

    print(' '.join(output))