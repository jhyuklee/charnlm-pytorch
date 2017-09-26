import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable


def run_epoch(m, d, mode='tr', is_train=True):
    running_loss = total_loss = 0.0
    print_step = 100
    run_step = total_step = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(m.parameters(), lr=m.config.lr)

    while True:
        optimizer.zero_grad()
        inputs, targets, lengths = d.get_next_batch(mode=mode, pad=True)
        inputs, targets, lengths = (
                Variable(torch.LongTensor(inputs).cuda()),
                Variable(torch.LongTensor(targets).cuda()),
                Variable(torch.LongTensor(lengths).cuda()))
        
        if is_train:
            m.train()
        else:
            m.eval()

        m.hidden = m.init_hidden(inputs.size(0))
        outputs = m(inputs)
        loss = criterion(outputs, targets)
        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm(m.parameters(), m.config.rnn_max_norm)
            optimizer.step()

        running_loss += np.exp(loss.data[0])
        run_step += 1.0
        total_loss += np.exp(loss.data[0])
        total_step += 1.0

        if (d.get_batch_ptr(mode)) % (m.config.batch_size * print_step) == 0:
            print('[%d] loss: %.3f batch_size: %d' % (d.get_batch_ptr(mode), 
                        running_loss / run_step, inputs.size(0)))
            run_step = 0
            running_loss = 0
        
        if d.get_batch_ptr(mode) == 0:
            print('total loss: %.3f\n' % (total_loss / total_step))
            return total_loss / total_step 

