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
        self.char_embed = nn.Embedding(config.char_vocab_size, config.char_embed_dim,
                padding_idx=1)
        self.char_conv = nn.ModuleList([nn.Conv2d(
                config.char_embed_dim, config.char_conv_fn[i],
                (config.char_conv_fh[i], config.char_conv_fw[i]),
                stride=1) for i in range(len(config.char_conv_fn))])
        self.input_dim = int(np.sum(config.char_conv_fn)) 
        self.lstm = nn.LSTM(self.input_dim, config.hidden_dim, config.layer_num,
                dropout=config.rnn_dr, batch_first=True)

        self.hw_nonl = nn.Linear(self.input_dim, self.input_dim)
        self.hw_gate = nn.Linear(self.input_dim, self.input_dim)
        self.fc1 = nn.Linear(config.hidden_dim, config.word_vocab_size)

        self.init_weights()
        self.hidden = self.init_hidden(config.batch_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=config.lr)
        self.model_params(debug=False)
    
    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(
                    self.config.layer_num, batch_size, self.config.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(
                    self.config.layer_num, batch_size, self.config.hidden_dim)).cuda())

    def init_weights(self):
        # self.char_embed.weight.data.uniform_(-0.05, 0.05)
        # for conv in self.char_conv:
        #     conv.weight.data.uniform_(-0.05, 0.05)
        #     conv.bias.data.fill_(0)
        # self.hw_nonl.weight.data.uniform_(-0.05, 0.05)
        # self.hw_nonl.bias.data.fill_(0)
        # self.hw_gate.weight.data.uniform_(-0.05, 0.05)
        self.hw_gate.bias.data.uniform_(-2.05, 2.05)
        # self.fc1.weight.data.uniform_(-0.05, 0.05)
        # self.fc1.bias.data.fill_(0)

    def model_params(self, debug=True):
        print('### model parameters')
        params = []
        total_size = 0
        def multiply_iter(p_list):
            out = 1
            for p in p_list:
                out *= p
            return out

        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
                total_size += multiply_iter(p.size())
            if debug:
                print(p.requires_grad, p.size())
        print('total size: %s\n' % '{:,}'.format(total_size))
        return params

    def char_conv_layer(self, inputs):
        embeds = self.char_embed(inputs.view(-1, self.config.max_wordlen))
        embeds = torch.transpose(torch.unsqueeze(embeds, 2), 1, 3).contiguous()
        conv_result = []
        for i, conv in enumerate(self.char_conv):
            char_conv = torch.squeeze(conv(embeds))
            char_mp = torch.max(torch.tanh(char_conv), 2)[0]
            char_mp = char_mp.view(-1, inputs.size(1), char_mp.size(1))
            conv_result.append(char_mp)
        conv_result = torch.cat(conv_result, 2)
        return conv_result

    def rnn_layer(self, inputs, hidden):
        lstm_out, hidden = self.lstm(inputs, hidden)
        return lstm_out.contiguous().view(-1, self.config.hidden_dim), hidden

    def highway_layer(self, inputs):
        nonl = F.relu(self.hw_nonl(inputs))
        gate = F.sigmoid(self.hw_gate(inputs))
        z = torch.mul(gate, nonl) + torch.mul(1-gate, inputs)
        return z

    def forward(self, inputs, hidden):
        char_conv = self.char_conv_layer(inputs)
        high_out = self.highway_layer(char_conv)
        out, hidden = self.rnn_layer(high_out, hidden)
        outputs = self.fc1(out)
        return outputs, hidden

    def decay_lr(self):
        self.config.lr /= 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.lr

    def save_checkpoint(self, state, is_best, filename=None):
        if filename is None:
            filename = self.config.checkpoint_path
        # print('### save checkpoint %s' % filename)
        torch.save(state, filename)

    def load_checkpoint(self, filename=None):
        if filename is None:
            filename = self.config.load_path
        print('### load checkpoint %s' % filename)
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    # deprecated but can be useful
    def create_mask(self, lengths, max_length):
        r = torch.unsqueeze(torch.arange(0, max_length), 0).long().cuda() # (1, 82)
        l = torch.unsqueeze(lengths, 1).expand(lengths.size(0), max_length) # (20, 82)
        mask = torch.lt(r.expand_as(l), l)
        return mask

