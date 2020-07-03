import torch
import torch.nn as nn

class Network(nn.Module):

    def __init__(self, vocab_size, embed_size, label_size, dropout):
        super(Network, self).__init__()
        self.vocab_size =  vocab_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.label_size = label_size
        self.lstm_hidden_size = embed_size

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=2,
                            dropout=self.dropout,
                            bidirectional=True)
        self.fc1 = nn.Linear(self.lstm_hidden_size*2, self.lstm_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self, init_range=0.5):
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.fc1.weight.data.uniform_(-init_range, init_range)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.weight.data.uniform_(-init_range, init_range)
        self.fc2.bias.data.fill_(0.0)

    def forward(self, text, text_length):
        embeded = self.embed(text)
        packed = nn.utils.rnn.pack_padded_sequence(embeded, text_length)
        packed_output, (hidden, cell) = self.lstm(packed)

        #combine both direction of the last lstm layer
        combined = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        combined = combined.squeeze(0)

        logits = self.relu(self.fc1(combined))

        logits = self.sigmoid(self.fc2(logits))

        return logits



