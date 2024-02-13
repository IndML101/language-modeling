from src.train.trainers import TransformerTrainer, TransformerDecoderTrainer, GPTTrainer
from src.dataset.preprocess import WikiTextDataProcessor, PTBDataProcessor


# To start training
def ptb_transformer_train():
    # vdim must be equal to d_model//n_heads
    model_args = dict(
        n_layers=6,
        d_model=256,
        max_seq_length=35,
        dff=1024,
        n_heads=8,
        dropout=0.1,
        vdim=32,
        bias=True,
        epsilon=1e-5,
    )
    TransformerTrainer(
        text_processor=PTBDataProcessor(),
        learning_rate=5.0,
        batch_size=40,
        decay_steps=1.0,
        **model_args,
    ).run(100)


# To continue training from last checkpoint
def ptb_transformer_retrain(tstamp: str):
    model_args = dict(
        n_layers=6,
        d_model=256,
        max_seq_length=35,
        dff=1024,
        n_heads=8,
        dropout=0.1,
        vdim=32,
        bias=True,
        epsilon=1e-5,
    )
    model_name = "Artifacts/Transformer-n{}-d{}-dff{}-h{}-{}.pt".format(
        model_args["n_layers"],
        model_args["d_model"],
        model_args["dff"],
        model_args["n_heads"],
        tstamp,
    )
    text_processor = "Artifacts/Transformer-n{}-d{}-dff{}-h{}-{}.dt".format(
        model_args["n_layers"],
        model_args["d_model"],
        model_args["dff"],
        model_args["n_heads"],
        tstamp,
    )
    last_val_loss = 3.48
    TransformerTrainer(
        model=model_name,
        text_processor=PTBDataProcessor(),
        learning_rate=5.0,
        batch_size=40,
        decay_steps=20.0,
        **model_args,
    ).run(200, last_val_loss=last_val_loss)


# To start training
def wikitext2_transformer_train():
    # vdim must be equal to d_model//n_heads
    model_args = dict(
        n_layers=4,
        d_model=256,
        max_seq_length=35,
        dff=1024,
        n_heads=8,
        dropout=0.1,
        vdim=32,
        bias=True,
        epsilon=1e-5,
    )
    TransformerTrainer(
        learning_rate=5.0, batch_size=40, decay_steps=1.0, **model_args
    ).run(100)


# To continue training from last checkpoint
def wikitext2_transformer_retrain(tstamp: str):
    model_args = dict(
        n_layers=6,
        d_model=256,
        max_seq_length=35,
        dff=1024,
        n_heads=8,
        dropout=0.1,
        vdim=32,
        bias=True,
        epsilon=1e-5,
    )
    model_name = "Artifacts/Transformer-n{}-d{}-dff{}-h{}-{}.pt".format(
        model_args["n_layers"],
        model_args["d_model"],
        model_args["dff"],
        model_args["n_heads"],
        tstamp,
    )
    text_processor = "Artifacts/Transformer-n{}-d{}-dff{}-h{}-{}.dt".format(
        model_args["n_layers"],
        model_args["d_model"],
        model_args["dff"],
        model_args["n_heads"],
        tstamp,
    )
    last_val_loss = 7.16
    TransformerTrainer(
        model=model_name,
        text_processor=WikiTextDataProcessor(),
        learning_rate=5.0,
        batch_size=40,
        decay_steps=1.0,
        **model_args,
    ).run(100, last_val_loss=last_val_loss)


# To start training
def ptb_pre_trained_decoder_train():
    # vdim must be equal to d_model//n_heads
    model_args = dict(
        n_layers=6,
        d_model=256,
        max_seq_length=35,
        dff=1024,
        n_heads=8,
        dropout=0.1,
        vdim=32,
        bias=True,
        epsilon=1e-5,
    )
    TransformerDecoderTrainer(
        text_processor=PTBDataProcessor(),
        learning_rate=5.0,
        batch_size=45,
        decay_steps=1.0,
        **model_args,
    ).run(100)


# To continue training from last checkpoint uncomment this line
def ptb_pre_trained_decoder_retrain(tstamp: str):
    model_args = dict(
        n_layers=6,
        d_model=256,
        max_seq_length=35,
        dff=1024,
        n_heads=8,
        dropout=0.1,
        vdim=32,
        bias=True,
        epsilon=1e-5,
    )
    model_name = "Artifacts/PreTrainedDecoder-n{}-d{}-dff{}-h{}-{}.pt".format(
        model_args["n_layers"],
        model_args["d_model"],
        model_args["dff"],
        model_args["n_heads"],
        tstamp,
    )
    text_processor = "Artifacts/PreTrainedDecoder-n{}-d{}-dff{}-h{}-{}.dt".format(
        model_args["n_layers"],
        model_args["d_model"],
        model_args["dff"],
        model_args["n_heads"],
        tstamp,
    )
    last_val_loss = 6.39
    TransformerTrainer(
        model=model_name,
        text_processor=PTBDataProcessor(),
        learning_rate=1e-5,
        batch_size=64,
        decay_steps=20.0,
        **model_args,
    ).run(200, last_val_loss=last_val_loss)


# To start training
def ptb_gpt_train():
    # vdim must be equal to d_model//n_heads
    model_args = dict(
        n_layers=12,
        d_model=512,
        max_seq_length=10,
        dff=2048,
        n_heads=8,
        dropout=0.1,
        vdim=64,
        bias=True,
        epsilon=1e-5,
    )
    GPTTrainer(
        text_processor=PTBDataProcessor(),
        learning_rate=5.0,
        batch_size=40,
        decay_steps=1.0,
        **model_args,
    ).run(100)


# To continue training from last checkpoint
def ptb_gpt_retrain(tstamp: str):
    model_args = dict(
        n_layers=12,
        d_model=512,
        max_seq_length=10,
        dff=2048,
        n_heads=8,
        dropout=0.1,
        vdim=64,
        bias=True,
        epsilon=1e-5,
    )
    model_name = "Artifacts/Transformer-n{}-d{}-dff{}-h{}-{}.pt".format(
        model_args["n_layers"],
        model_args["d_model"],
        model_args["dff"],
        model_args["n_heads"],
        tstamp,
    )
    text_processor = "Artifacts/Transformer-n{}-d{}-dff{}-h{}-{}.dt".format(
        model_args["n_layers"],
        model_args["d_model"],
        model_args["dff"],
        model_args["n_heads"],
        tstamp,
    )
    last_val_loss = 3.48
    GPTTrainer(
        model=model_name,
        text_processor=PTBDataProcessor(),
        learning_rate=5.0,
        batch_size=40,
        decay_steps=20.0,
        **model_args,
    ).run(100, last_val_loss=last_val_loss)


if __name__ == "__main__":
    tstamp = ""

    # wikitext2_transformer_train()
    # wikitext2_transformer_retrain(tstamp)

    # ptb_transformer_train()
    # ptb_transformer_retrain(tstamp)

    # ptb_pre_trained_decoder_train()
    # ptb_pre_trained_decoder_retrain(tstamp)

    ptb_gpt_train()
    # ptb_gpt_retrain(tstamp)
