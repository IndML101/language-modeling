The aim of this repository is to train Transformer architecture based language models from scratch.
All the scripts are organised in the `src` folder:
```
src
├── base
│   ├── __init__.py
│   ├── modules.py
│   └── __pycache__
├── dataset
│   ├── batches.py
│   ├── __init__.py
│   ├── loaders.py
│   ├── preprocess.py
│   └── __pycache__
├── __init__.py
├── models
│   ├── __init__.py
│   ├── llms.py
│   └── __pycache__
├── modules
│   ├── __init__.py
│   ├── __pycache__
│   └── utils.py
├── __pycache__
│   └── __init__.cpython-311.pyc
└── train
    ├── __init__.py
    ├── __pycache__
    ├── trainers.py
    └── utils.py
```
- `base` folder has the abstract classes.
- `dataset` folder has the logic for data loaders and preprocessers (vocabulary and tokenization).
- `models` folder has the logic for different model classes.
- `modules` folder has the logic for components for individual models like Attention, Encoder, Decoder etc.
- `train` folder has training scripts.

Here is the snippet of the training logs:
```
| epoch  21 |   100/  500 batches | lr 3.58e-06 | ms/batch 273.46 | loss 10.37 | ppl 31807.92
| epoch  21 |   200/  500 batches | lr 3.58e-06 | ms/batch 271.20 | loss 10.26 | ppl 28701.46
| epoch  21 |   300/  500 batches | lr 3.58e-06 | ms/batch 271.03 | loss 10.26 | ppl 28697.14
| epoch  21 |   400/  500 batches | lr 3.58e-06 | ms/batch 271.30 | loss 10.26 | ppl 28693.66
| epoch  21 |   500/  500 batches | lr 3.58e-06 | ms/batch 269.39 | loss 10.26 | ppl 28689.20
-----------------------------------------------------------------------------------------
| end of epoch  21 | time: 143.28s | valid loss  7.19 | valid ppl  1330.17
-----------------------------------------------------------------------------------------
| epoch  22 |   100/  500 batches | lr 3.41e-06 | ms/batch 273.62 | loss 10.37 | ppl 31784.38
| epoch  22 |   200/  500 batches | lr 3.41e-06 | ms/batch 271.17 | loss 10.26 | ppl 28679.80
| epoch  22 |   300/  500 batches | lr 3.41e-06 | ms/batch 271.27 | loss 10.26 | ppl 28674.66
| epoch  22 |   400/  500 batches | lr 3.41e-06 | ms/batch 270.92 | loss 10.26 | ppl 28670.64
| epoch  22 |   500/  500 batches | lr 3.41e-06 | ms/batch 269.12 | loss 10.26 | ppl 28665.42
-----------------------------------------------------------------------------------------
| end of epoch  22 | time: 143.18s | valid loss  7.19 | valid ppl  1329.31
-----------------------------------------------------------------------------------------
| epoch  23 |   100/  500 batches | lr 3.24e-06 | ms/batch 273.37 | loss 10.37 | ppl 31756.65
| epoch  23 |   200/  500 batches | lr 3.24e-06 | ms/batch 270.62 | loss 10.26 | ppl 28654.41
| epoch  23 |   300/  500 batches | lr 3.24e-06 | ms/batch 271.25 | loss 10.26 | ppl 28648.42
| epoch  23 |   400/  500 batches | lr 3.24e-06 | ms/batch 270.60 | loss 10.26 | ppl 28643.90
| epoch  23 |   500/  500 batches | lr 3.24e-06 | ms/batch 269.27 | loss 10.26 | ppl 28637.89
-----------------------------------------------------------------------------------------
| end of epoch  23 | time: 143.17s | valid loss  7.19 | valid ppl  1328.33
-----------------------------------------------------------------------------------------
| epoch  24 |   100/  500 batches | lr 3.07e-06 | ms/batch 271.09 | loss 10.36 | ppl 31724.68
| epoch  24 |   200/  500 batches | lr 3.07e-06 | ms/batch 269.04 | loss 10.26 | ppl 28625.24
| epoch  24 |   300/  500 batches | lr 3.07e-06 | ms/batch 269.76 | loss 10.26 | ppl 28618.39
| epoch  24 |   400/  500 batches | lr 3.07e-06 | ms/batch 270.33 | loss 10.26 | ppl 28613.38
| epoch  24 |   500/  500 batches | lr 3.07e-06 | ms/batch 267.69 | loss 10.26 | ppl 28606.58
-----------------------------------------------------------------------------------------
| end of epoch  24 | time: 142.31s | valid loss  7.19 | valid ppl  1327.23
-----------------------------------------------------------------------------------------
| epoch  25 |   100/  500 batches | lr 2.92e-06 | ms/batch 272.53 | loss 10.36 | ppl 31688.41
| epoch  25 |   200/  500 batches | lr 2.92e-06 | ms/batch 268.16 | loss 10.26 | ppl 28592.26
| epoch  25 |   300/  500 batches | lr 2.92e-06 | ms/batch 270.39 | loss 10.26 | ppl 28584.47
| epoch  25 |   400/  500 batches | lr 2.92e-06 | ms/batch 269.74 | loss 10.26 | ppl 28579.03
| epoch  25 |   500/  500 batches | lr 2.92e-06 | ms/batch 268.20 | loss 10.26 | ppl 28571.45
-----------------------------------------------------------------------------------------
| end of epoch  25 | time: 142.48s | valid loss  7.19 | valid ppl  1326.01
-----------------------------------------------------------------------------------------
```