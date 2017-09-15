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
        inputs, targets = d.get_next_batch(mode=mode)
        inputs, targets = (torch.LongTensor(inputs).cuda(), 
                torch.LongTensor(targets).cuda())
        inputs, targets = Variable(inputs), Variable(targets)

        m.hidden = m.init_hidden(inputs.size(0))
        outputs = m(inputs)
        targets = targets.view(-1)
        
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

