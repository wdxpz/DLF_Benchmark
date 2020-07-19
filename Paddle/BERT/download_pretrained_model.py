import os
from paddlepalm import downloader

from .bert import model_dir

downloader.download('pretrain', 'BERT-en-uncased-base', os.path.dirname(model_dir))