import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import torch.nn.functional as F

from config import TextClassifier_Config
from network import Network

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_FILE = 'text_bilstm_checkpoint_{}.pth'
RESULT_DIR = os.path.join(BASE_DIR, 'result')
RESULT_FILE = os.path.join(RESULT_DIR, 'text_bilstm_result.txt')

RANDOM_SEED = 123

class TextClassifier(object):

    def __init__(self, config=TextClassifier_Config):
        super(TextClassifier, self).__init__()

        self.works = config['workers']
        self.ngpu = config['ngpu']
        self.vocab_size = config['vocabulary_size']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.train_valid_ration = config['train_valid_ratio']
        self.dropout = config['dropout']
        self.lr = config['lr']
        self.embed_size = config['embed_size']
        self.print_every = config['print_every']

        #device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #data
        self.train_data, self.test_data, self.vocab_size, self.label_size = self._init_data()
        #network
        self.net = self._init_network()

    def _init_network(self):
        net = Network(self.vocab_size, self.embed_size, 1, self.dropout)
        net.to(self.device)

        if self.device.type == 'cuda' and self.ngpu > 1:
            net = nn.DataParallel(net, list(range(self.ngpu)))

        return net

    def _init_data(self):

        TEXT = torchtext.data.Field(lower=True, include_lengths=True)  # necessary for packed_padded_sequence
        LABEL = torchtext.data.LabelField(dtype=torch.float)
        train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL, root=DATA_DIR)
        # train_data, valid_data = org_train_data.split(random_state=random.seed(RANDOM_SEED), split_ratio=0.8)

        print(f'Num Train: {len(train_data)}')
        # print(f'Num Valid: {len(valid_data)}')
        print(f'Num Test: {len(test_data)}')

        TEXT.build_vocab(train_data, max_size=self.vocab_size)
        LABEL.build_vocab(train_data)

        print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
        print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
        # %%
        train_loader, test_loader = torchtext.data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=self.batch_size,
            sort_within_batch=True,  # necessary for packed_padded_sequence
            device=self.device)

        return train_loader, test_loader, len(TEXT.vocab), len(LABEL.vocab)

    def train(self):
        criterion = nn.BCELoss().to(self.device)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        self.net.train()

        train_accu = 0
        train_loss = 0
        start_time = time.time()

        for epoch in range(1, self.num_epochs+1):
            tsampels = 0
            for i, batch in enumerate(self.train_data):
                loss, accu, nsamples = self._train_func(batch, criterion, optimizer)
                train_loss += loss
                train_accu += accu
                tsampels += nsamples

                if (i+1) % self.print_every == 0 or (i+1) == len(self.train_data):
                    print('epoch:[{}/{}] step:[{}/{}]\ttrain_loss: {:.4f}\ttrain_accu: {:.4f}'.format(
                        epoch, self.num_epochs, i+1, len(self.train_data),
                        train_loss/tsampels, train_accu/tsampels))
                    train_loss = 0
                    train_accu = 0
                    tsampels = 0

        # scheduler.step()

        total_time = time.time() - start_time
        with open(RESULT_FILE, 'a') as f:
            f.write(f'\ntraining results: \n')
            f.write('\n total training time: \t {}'.format(total_time))
            # f.write('\n final average training lostt: \t {:.3f}'.format(train_loss))
            # f.write('\n final average training accuracy: \t{:.3f}'.format(train_accu))


    def _train_func(self, batch, criterion, optimizer):
        self.net.train()
        optimizer.zero_grad()

        text, text_length, labels = batch.text[0], batch.text[1], batch.label
        # text, text_length, labels = text.to(self.device), text_length.to(self.device), labels.to(self.device)

        logits = self.net(text, text_length)
        loss = criterion(logits, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

        predicts = (logits>0.5).long()
        accu = (predicts.squeeze(1) == labels).sum().item()
        nsamples = labels.size()[0]

        return loss.item(), accu, nsamples

    def test(self):
        self.net.eval()

        correct = 0
        total = 0

        start_time = time.time()
        with torch.no_grad():
            for i, batch in enumerate(self.test_data):
                text, text_length, labels = batch.text[0], batch.text[1], batch.label
                # text, text_length, labels = text.to(self.device), text_length.to(self.device), labels.to(self.device)

                logits = self.net(text, text_length)
                predicts = (logits > 0.5).long()
                correct += (predicts.squeeze(1) == labels).sum().item()
                total += labels.size()[0]
        total_time = time.time() - start_time

        s = '\ntesting results: \n accuracy on {} test reviews: {:.3f}, total time: {} s\n'.format(total,
                                                                                                  correct * 1.0 / total,
                                                                                                  total_time)
        print(s)
        with open(RESULT_FILE, 'a') as f:
            f.write(s)






