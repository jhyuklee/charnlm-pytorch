import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim


class Char_NLM(nn.Module):
    def __init__(self, config):
        super(Char_NLM, self).__init__()
        self.config = config
        self.char_conv1 = nn.Conv2d(
                config.char_embed_dim,
                config.conv1_filter_num,
                (config.conv1_filter_width, config.char_embed_dim),
                stride=1)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.1)
        print()

    def forward(self, input):
        print(input)
        input = self.char_conv1(input)
        print(input)
        return input

    
