import os


TextClassifier_Config = {
    'workers': 1,
    'ngpu': 1,
    'vocabulary_size': 32650,
    'train_valid_ratio': 0.8,
    'dropout': 0,
    'embed_size': 64,
    'lr': 0.0001,
    'num_epochs': 20,
    'batch_size': 64,
    'print_every': 100,
    'save_every': 10000
}