import torch
import argparse

from torch.autograd import Variable
from dataset import Dataset
from model import Char_NLM


argparser = argparse.ArgumentParser()
argparser.add_argument('--train_path', type=str, default='./data/train.txt')
argparser.add_argument('--valid_path', type=str, default='./data/valid.txt')
argparser.add_argument('--test_path', type=str, default='./data/test.txt')
argparser.add_argument('--batch_size', type=int, default=20)
argparser.add_argument('--epoch', type=int, default=10)
argparser.add_argument('--train', action='store_true', default=True)
argparser.add_argument('--valid', action='store_true', default=True)
argparser.add_argument('--test', action='store_true', default=True)

argparser.add_argument('--char_embed_dim', type=int, default=10)
argparser.add_argument('--conv1_filter_num', type=int, default=5)
argparser.add_argument('--conv1_filter_width', type=int, default=3)
args = argparser.parse_args()


def run_experiment(model, dataset):
    if model.config.train:
        print('##### Training #####')
        for ep in range(model.config.epoch):
            while True:
                inputs, targets = dataset.get_next_batch()
                inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
                inputs, targets = Variable(inputs), Variable(targets)
                model.optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]
                if model.train_ptr % (model.config.batch_size * 5) == 0:
                    print('[%d, %5d] loss: %.3f' % (ep + 1, i + 1, 
                                running_loss / model.config.batch_size * 5))
                    running_loss = 0.0



        if model.config.valid:
            print('##### Validation #####')
    
    if model.config.test:
        print('##### Testing #####')


def main():
    print('##### Dataset Loading #####')
    dataset = Dataset(args)

    print('##### Model Loading #####')
    model = Char_NLM(args)
    run_experiment(model, dataset)


if __name__ == '__main__':
    main()
