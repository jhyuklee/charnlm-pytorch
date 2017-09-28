import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable


def run_epoch(m, d, mode='tr', is_train=True):
    running_loss = total_loss = 0.0
    print_step = 10
    run_step = total_step = 0.0

    while True:
        m.optimizer.zero_grad()
        inputs, targets = d.get_next_batch(m.config.seq_len, mode=mode)
        inputs, targets = (
                Variable(torch.LongTensor(inputs).cuda()),
                Variable(torch.LongTensor(targets).cuda()))
        
        if is_train:
            m.train()
        else:
            m.eval()

        m.hidden = [state.detach() for state in m.hidden]
        outputs, m.hidden = m(inputs, m.hidden)
        loss = m.criterion(outputs, targets.view(-1))
        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm(m.parameters(), m.config.rnn_max_norm)
            m.optimizer.step()

        running_loss += np.exp(loss.data[0])
        run_step += 1.0
        total_loss += np.exp(loss.data[0])
        total_step += 1.0

        if (d.get_batch_ptr(mode)) % (m.config.batch_size * print_step) == 0:
            print('[%d] loss: %.3f seq_len: %d' % (d.get_batch_ptr(mode), 
                        running_loss / run_step, inputs.size(1)))
            run_step = 0
            running_loss = 0
            if m.config.save:
                m.save_checkpoint({
                    'config': m.config,
                    'state_dict': m.state_dict(),
                    'optimizer': m.optimizer.state_dict(),
                }, False)

        
        if d.get_batch_ptr(mode) == 0:
            print('total loss: %.3f\n' % (total_loss / total_step))
            return total_loss / total_step 

