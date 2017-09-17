import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys


class Char_NLM(nn.Module):
    def __init__(self, config):
        super(Char_NLM, self).__init__()
        self.config = config
        self.char_embed = nn.Embedding(config.char_vocab_size, config.char_embed_dim)
        self.char_conv1 = nn.Conv2d(
                1, config.conv1_filter_num,
                (config.conv1_filter_width, config.char_embed_dim),
                stride=1)
        self.lstm = nn.LSTM(config.conv1_filter_num, config.hidden_dim)
        self.linear1 = nn.Linear(config.hidden_dim, config.word_vocab_size)
    
    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(
                    1, batch_size, self.config.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(
                    1, batch_size, self.config.hidden_dim)).cuda())

    def create_mask(self, lengths, max_length):
        r = torch.unsqueeze(torch.arange(0, max_length), 0).long().cuda() # (1, 82)
        # print(r)
        l = torch.unsqueeze(lengths, 1).expand(lengths.size(0), max_length) # (20, 82)
        # print(l)
        mask = torch.lt(r.expand_as(l), l)
        # print(mask)
        return mask

    def forward(self, inputs):
        # print(inputs.size())
        embeds = self.char_embed(inputs.view(-1, self.config.max_wordlen))
        # print(torch.unsqueeze(embeds, 1).size())
        conv1 = torch.squeeze(self.char_conv1(torch.unsqueeze(embeds, 1)))
        # print(conv1.size())
        maxpool = torch.max(torch.tanh(conv1), 2)[0]
        # print(maxpool.size())
        maxpool = maxpool.view(-1, self.config.max_sentlen, maxpool.size(1))
        # print(maxpool.size())
        lstm_out, self.hidden = self.lstm(torch.transpose(maxpool, 0, 1),
                self.hidden)
        # print(lstm_out.size(), self.hidden[0].size())
        outputs = self.linear1(lstm_out.view(-1, self.config.hidden_dim))
        outputs_t = torch.transpose(outputs.view(
            self.config.max_sentlen, -1, self.config.word_vocab_size),
            0, 1).contiguous()
        # print(outputs_t.size())
        
        return outputs_t.view(-1, self.config.word_vocab_size)

