import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable


def run_epoch(m, d, mode='tr', is_train=True):
    running_loss = 0.0
    print_step = 100
    run_step = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(m.parameters(), lr=0.1)

    while True:
        optimizer.zero_grad()
        inputs, targets, lengths = d.get_next_batch(mode=mode)
        inputs, targets, lengths = (
                Variable(torch.LongTensor(inputs).cuda()),
                Variable(torch.LongTensor(targets).cuda()),
                Variable(torch.LongTensor(lengths).cuda()))
        
        mask = m.create_mask(lengths-1, m.config.max_sentlen) # adjust targets later
        m.hidden = m.init_hidden(inputs.size(0))
        outputs = m(inputs)
        targets = targets.view(-1)
        o_mask = torch.unsqueeze(mask.view(-1), 1).expand_as(outputs)
        outputs = torch.masked_select(outputs, o_mask).view(
                -1, m.config.word_vocab_size)
        targets = torch.masked_select(targets, mask.view(-1))
        loss = criterion(outputs, targets)
        
        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += np.exp(loss.data[0])
        run_step += 1.0
        if (d.get_batch_ptr(mode)) % (m.config.batch_size * print_step) == 0:
            print('[%d] loss: %.3f batch_size: %d' % (d.get_batch_ptr(mode), 
                        running_loss / run_step, inputs.size(0)))
        
        if d.get_batch_ptr(mode) == 0:
            print('total loss: %.3f\n' % (running_loss / run_step))
            break

