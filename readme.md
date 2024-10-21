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