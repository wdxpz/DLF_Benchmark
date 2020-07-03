import os


TextClassifier_Config = {
    'workers': 1,
    'ngpu': 1,
    'bert_model': 'bert-base-uncased', #'bert-large-uncased',
    'max_sentence_length': 128,
    'dropout': 0,
    'embed_size': 16,
    'lr': 0.00001,
    'epsilon': 1e-5,
    'num_epochs': 3,
    'batch_size': 32,
    'print_every': 100,
    'save_every': 10000
}