import torch
import argparse
import pickle
import pprint
from torch.autograd import Variable
from dataset import Dataset, Config
from model import Char_NLM
from run import run_epoch


argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, default='./data/preprocess(tmp).pkl')
argparser.add_argument('--batch_size', type=int, default=20)
argparser.add_argument('--epoch', type=int, default=25)
argparser.add_argument('--train', action='store_true', default=True)
argparser.add_argument('--valid', action='store_true', default=True)
argparser.add_argument('--test', action='store_true', default=True)

argparser.add_argument('--hidden_dim', type=int, default=300)
argparser.add_argument('--layer_num', type=int, default=2)
argparser.add_argument('--char_embed_dim', type=int, default=15)
argparser.add_argument('--char_conv_fn', type=list, 
        default=[25, 50, 75, 100, 125, 150])
argparser.add_argument('--char_conv_fh', type=list, default=[1, 1, 1, 1, 1, 1])
argparser.add_argument('--char_conv_fw', type=list, default=[1, 2, 3, 4, 5, 6])
args = argparser.parse_args()


def run_experiment(model, dataset):
    if model.config.train:
        print('##### Training #####')
        for ep in range(model.config.epoch):
            print('[Epoch %d]' % (ep+1))
            run_epoch(model, dataset, 'tr')

            if model.config.valid:
                print('##### Validation #####')
                run_epoch(model, dataset, 'va', is_train=False)
    
    if model.config.test:
        print('##### Testing #####')
        run_epoch(model, dataset, 'te', is_train=False)


def main():
    print('### load dataset')
    dataset = pickle.load(open(args.data_path, 'rb'))
    
    # update args
    dataset.config.__dict__.update(args.__dict__)
    args.__dict__.update(dataset.config.__dict__)
    pp = lambda x: pprint.PrettyPrinter().pprint(x)
    pp(vars(dataset.config))
    print()

    model = Char_NLM(args).cuda()
    run_experiment(model, dataset)


if __name__ == '__main__':
    main()

