import torch
import argparse
from torch.autograd import Variable
from dataset import Dataset
from model import Char_NLM
from run import run_epoch


argparser = argparse.ArgumentParser()
argparser.add_argument('--train_path', type=str, default='./data/train.txt')
argparser.add_argument('--valid_path', type=str, default='./data/valid.txt')
argparser.add_argument('--test_path', type=str, default='./data/test.txt')
argparser.add_argument('--batch_size', type=int, default=20)
argparser.add_argument('--epoch', type=int, default=2)
argparser.add_argument('--train', action='store_true', default=True)
argparser.add_argument('--valid', action='store_true', default=True)
argparser.add_argument('--test', action='store_true', default=True)

argparser.add_argument('--hidden_dim', type=int, default=10)
argparser.add_argument('--max_wordlen', type=int, default=0)
argparser.add_argument('--max_sentlen', type=int, default=0)
argparser.add_argument('--word_vocab_size', type=int, default=0)
argparser.add_argument('--char_vocab_size', type=int, default=0)
argparser.add_argument('--char_embed_dim', type=int, default=10)
argparser.add_argument('--conv1_filter_num', type=int, default=5)
argparser.add_argument('--conv1_filter_width', type=int, default=3)
args = argparser.parse_args()


def run_experiment(model, dataset):
    if model.config.train:
        print('##### Training #####')
        for ep in range(model.config.epoch):
            print('[Epoch %d]' % ep)
            run_epoch(model, dataset, 'tr')

            if model.config.valid:
                print('##### Validation #####')
                run_epoch(model, dataset, 'va')
    
    if model.config.test:
        print('##### Testing #####')
        run_epoch(model, dataset, 'te')


def main():
    print('##### Dataset Loading #####')
    dataset = Dataset(args)

    model = Char_NLM(args).cuda()
    run_experiment(model, dataset)


if __name__ == '__main__':
    main()

