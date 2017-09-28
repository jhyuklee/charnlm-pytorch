import numpy as np
import sys
import nltk
import pickle
import pprint
import copy

nltk.download('punkt')

class Dataset(object):
    def __init__(self, config):
        self.config = config

        # dictionary settings
        self.initialize_dictionary()
        self.build_corpus(self.config.train_path, self.train_corpus)
        self.build_corpus(self.config.valid_path, self.valid_corpus)
        self.build_corpus(self.config.test_path, self.test_corpus)
        print()

        self.train_data = self.process_data(
                self.train_corpus, 
                update_dict=True)
        self.valid_data = self.process_data(
                self.valid_corpus,
                update_dict=True)
        self.test_data = self.process_data(
                self.test_corpus)
        print()

        self.pad_data(self.train_data)
        self.pad_data(self.valid_data)
        self.pad_data(self.test_data)

        self.train_data = self.reshape_data(self.train_data)
        self.valid_data = self.reshape_data(self.valid_data)
        self.test_data = self.reshape_data(self.test_data)
        print()

        self.train_ptr = 0
        self.valid_ptr = 0
        self.test_ptr = 0
        
        print('char_dict', len(self.char2idx))
        print('word_dict', len(self.word2idx), end='\n\n')

    def initialize_dictionary(self):
        self.train_corpus = []
        self.valid_corpus = []
        self.test_corpus = []
        self.char2idx = {}
        self.idx2char = {}
        self.word2idx = {}
        self.idx2word = {}
        self.UNK = '<unk>'
        self.PAD = 'PAD'
        self.CONJ = '+'
        self.START = '{'
        self.END = '}'
        self.char2idx[self.UNK] = 0
        self.char2idx[self.PAD] = 1
        self.char2idx[self.CONJ] = 2
        self.char2idx[self.START] = 3
        self.char2idx[self.END] = 4
        self.idx2char[0] = self.UNK
        self.idx2char[1] = self.PAD
        self.idx2char[2] = self.CONJ
        self.idx2char[3] = self.START
        self.idx2char[4] = self.END
        self.word2idx[self.UNK] = 0
        self.word2idx[self.PAD] = 1
        self.word2idx[self.CONJ] = 2
        self.idx2word[0] = self.UNK
        self.idx2word[1] = self.PAD
        self.idx2word[2] = self.CONJ
    
    def update_dictionary(self, key, mode=None):
        if mode == 'c':
            if key not in self.char2idx:
                self.char2idx[key] = len(self.char2idx)
                self.idx2char[len(self.idx2char)] = key
        elif mode == 'w':
            if key not in self.word2idx:
                self.word2idx[key] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = key
    
    def map_dictionary(self, key_list, dictionary, reverse=False):
        output = []
        # reverse=False : word2idx, char2idx
        # reverse=True : idx2word, idx2char
        for key in key_list:
            if key in dictionary:
                if reverse and key == 1: # PAD
                    continue
                else:
                    output.append(dictionary[key])
            else:
                if not reverse:
                    output.append(dictionary[self.UNK])
                else:
                    output.append(dictionary[0]) # 0 for UNK
        return output

    def build_corpus(self, path, corpus):
        print('building corpus %s' % path)
        with open(path) as f:
            for k, line in enumerate(f):
                # sentence_split = nltk.word_tokenize(line[:-1])
                sentence_split = line[:-1].split()
                for word in sentence_split:
                    corpus.append(word)
                corpus.append(self.CONJ)

    def process_data(self, corpus, update_dict=False):
        print('processing corpus %d' % len(corpus))
        total_data = []
        max_wordlen = 0

        for k, word in enumerate(corpus):
            # dictionary update
            if update_dict:
                self.update_dictionary(word, 'w')
                for char in word:
                    self.update_dictionary(char, 'c')
            
            # user special characters or mapping
            if word == self.UNK or word == self.CONJ or word == self.PAD:
                word_char = word
                charidx = [self.char2idx[word_char]]
            else:
                word_char = self.START + word + self.END
                charidx = self.map_dictionary(word_char, self.char2idx)
            
            # get max word length
            max_wordlen = (len(word_char) 
                    if len(word_char) > max_wordlen else max_wordlen)
            if max_wordlen > self.config.max_wordlen:
                self.config.max_wordlen = max_wordlen

            # word / char
            total_data.append([self.word2idx[word], charidx])

        if update_dict:
            self.config.char_vocab_size = len(self.char2idx)
            self.config.word_vocab_size = len(self.word2idx)

        print('data size', len(total_data))
        print('max wordlen', max_wordlen)

        return total_data

    def pad_data(self, dataset):
        for data in dataset:
            sentword, sentchar = data
            # pad word in sentchar
            while len(sentchar) != self.config.max_wordlen:
                sentchar.append(self.char2idx[self.PAD])
        return dataset

    def reshape_data(self, dataset):
        inputs = [d[1] for d in dataset]
        targets = [d[0] for d in dataset]
        seq_len = len(dataset) // self.config.batch_size
        inputs = np.array(inputs[:seq_len * self.config.batch_size])
        targets = np.array(targets[:seq_len * self.config.batch_size])

        inputs = np.reshape(inputs, (self.config.batch_size, seq_len, -1))
        targets = np.reshape(targets, (self.config.batch_size, -1))
        print('reshaped data', inputs.shape)
        return inputs, targets

    def get_next_batch(self, seq_len, mode='tr'):
        if mode == 'tr':
            ptr = self.train_ptr
            data = self.train_data
        elif mode == 'va':
            ptr = self.valid_ptr
            data = self.valid_data
        elif mode == 'te':
            ptr = self.test_ptr
            data = self.test_data
        
        seq_len = (seq_len if ptr + seq_len < len(data[0][0])
                else len(data[0][0]) - ptr - 1)
        inputs = data[0][:,ptr:ptr+seq_len,:]
        targets = data[1][:,ptr+1:ptr+seq_len+1]
        
        if len(data[0][0]) - (ptr + seq_len) == 1: # last batch
            ptr += 1

        if mode == 'tr':
            self.train_ptr = (ptr + seq_len) % len(data[0][0])
        elif mode == 'va':
            self.valid_ptr = (ptr + seq_len) % len(data[0][0])
        elif mode == 'te':
            self.test_ptr = (ptr + seq_len) % len(data[0][0])

        return inputs, targets
    
    def get_batch_ptr(self, mode):
        if mode == 'tr':
            return self.train_ptr
        elif mode == 'va':
            return self.valid_ptr
        elif mode == 'te':
            return self.test_ptr


class Config(object):
    def __init__(self):
        self.train_path = './data/train.txt'
        self.valid_path = './data/valid.txt'
        self.test_path = './data/test.txt'
        self.batch_size = 20
        self.max_wordlen = 0
        self.char_vocab_size = 0
        self.word_vocab_size = 0
        self.save_preprocess = True
        self.preprocess_save_path = './data/preprocess(tmp).pkl'
        self.preprocess_load_path = './data/preprocess(tmp).pkl'


if __name__ == '__main__':
    config = Config()
    if config.save_preprocess:
        dataset = Dataset(config)
        pickle.dump(dataset, open(config.preprocess_save_path, 'wb'))
    else:
        print('## load preprocess %s' % config.preprocess_load_path)
        dataset = pickle.load(open(config.preprocess_load_path, 'rb'))

    # dataset config must be valid
    pp = lambda x: pprint.PrettyPrinter().pprint(x)
    pp(([(k,v) for k, v in vars(dataset.config).items() if '__' not in k]))
    print()
    
    input, target = dataset.get_next_batch(seq_len=5)
    print([dataset.map_dictionary(i, dataset.idx2char) for i in input[0,:,:]])
    print([dataset.idx2word[t] for t in target[0,:]])
    print()

    input, target = dataset.get_next_batch(seq_len=5)
    print([dataset.map_dictionary(i, dataset.idx2char) for i in input[0,:,:]])
    print([dataset.idx2word[t] for t in target[0,:]])

    print('train', dataset.train_data[0].shape)
    print('valid', dataset.valid_data[0].shape)
    print('test', dataset.test_data[0].shape)
   
    while True:
        i, t = dataset.get_next_batch(seq_len=100, mode='te')
        print(dataset.test_ptr, len(i[0]))
        if dataset.test_ptr == 0:
            print('\niteration test pass!')
            break

