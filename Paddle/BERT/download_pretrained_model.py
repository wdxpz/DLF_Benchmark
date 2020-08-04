import os
from paddlepalm import downloader

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'model/bert_model/pretrain/BERT-en-uncased-base')

downloader.download('pretrain', 'BERT-en-uncased-base', os.path.dirname(model_dir))