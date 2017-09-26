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

        self.train_data = self.process_data(
                self.train_corpus, 
                update_dict=True)
        self.valid_data = self.process_data(
                self.valid_corpus,
                update_dict=True)
        self.test_data = self.process_data(
                self.test_corpus)

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
                    corpus.append(self.START + word + self.END)
                corpus.append(self.CONJ)

    def process_data(self, corpus, update_dict=False):
        print('processing corpus %d' % len(corpus))
        total_data = []
        max_wordlen = 0

        for k, word in enumerate(corpus):
            target = word
            sentence = []
            if k < self.config.max_sentlen:
                while len(sentence) != self.config.max_sentlen - k:
                    sentence.append(self.PAD)
                sentence = sentence + corpus[0:k]
                # print(len(sentence), sentence, target)
            else:
                sentence = corpus[k-self.config.max_sentlen:k]
                # print('here', len(sentence), sentence, target)

            # count max length
            sentence_char = ' '.join(sentence) 
            for word in sentence:
                max_wordlen = (len(word)
                        if len(word) > max_wordlen else max_wordlen)

            if update_dict:
                self.update_dictionary(target, 'w')
                for char in sentence_char:
                    self.update_dictionary(char, 'c')
                for word in sentence:
                    self.update_dictionary(word, 'w')
                if max_wordlen > self.config.max_wordlen:
                    self.config.max_wordlen = max_wordlen
            
            sentchar = []
            for word in sentence:
                if word == self.PAD:
                    sentchar.append([self.char2idx[self.PAD]])
                elif word == self.UNK:
                    sentchar.append([self.char2idx[self.UNK]]) 
                else:
                    sentchar.append(self.map_dictionary(word, self.char2idx))
            if target in self.word2idx:
                target = self.word2idx[target]
            else:
                target = self.word2idx[self.UNK]
            length = len(sentchar)

            assert len(sentchar) == self.config.max_sentlen
            total_data.append([sentchar, target, length])

        if update_dict:
            self.config.char_vocab_size = len(self.char2idx)
            self.config.word_vocab_size = len(self.word2idx)

        print('data size', len(total_data))
        print('max wordlen', max_wordlen, end='\n\n')

        return total_data

    def pad_data(self, dataset):
        for data in dataset:
            sentchar, sentword, _ = data
            # pad word in sentchar
            for word in sentchar:
                while len(word) != self.config.max_wordlen:
                    word.append(self.char2idx[self.PAD])

        return dataset
    
    def get_next_batch(self, mode='tr', batch_size=None, as_numpy=True, pad=True):
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if mode == 'tr':
            ptr = self.train_ptr
            data = self.train_data
        elif mode == 'va':
            ptr = self.valid_ptr
            data = self.valid_data
        elif mode == 'te':
            ptr = self.test_ptr
            data = self.test_data
        
        batch_size = (batch_size if ptr + batch_size <= len(data)
                else len(data) - ptr)
        if pad:
            padded_data = self.pad_data(copy.deepcopy(data[ptr:ptr+batch_size]))
        else:
            padded_data = data[ptr:ptr+batch_size]
        inputs = [d[0] for d in padded_data]
        targets = [d[1] for d in padded_data]
        lengths = [d[2] for d in padded_data]
        
        if mode == 'tr':
            self.train_ptr = (ptr + batch_size) % len(data)
        elif mode == 'va':
            self.valid_ptr = (ptr + batch_size) % len(data)
        elif mode == 'te':
            self.test_ptr = (ptr + batch_size) % len(data)

        if as_numpy:
            inputs = np.array(inputs)
            targets = np.array(targets)
            lengths = np.array(lengths)

        return inputs, targets, lengths
    
    def get_batch_ptr(self, mode):
        if mode == 'tr':
            return self.train_ptr
        elif mode == 'va':
            return self.valid_ptr
        elif mode == 'te':
            return self.test_ptr

    def shuffle_data(self, mode='tr', seed=None):
        if seed is not None:
            np.random.seed(seed)
        if mode == 'tr':
            np.random.shuffle(self.train_data)
        elif mode == 'va':
            np.random.shuffle(self.valid_data)
        elif mode == 'te':
            np.random.shuffle(self.test_data)


class Config(object):
    def __init__(self):
        self.train_path = './data/train.txt'
        self.valid_path = './data/valid.txt'
        self.test_path = './data/test.txt'
        self.batch_size = 32
        self.max_wordlen = 0
        self.max_sentlen = 35
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
    
    input, target, l = dataset.get_next_batch(batch_size=1)
    print([dataset.map_dictionary(i, dataset.idx2char) for i in input[0]])
    print(dataset.idx2word[target[0]])
    print()
    dataset.shuffle_data()
    input, target, l = dataset.get_next_batch(batch_size=1)
    print([dataset.map_dictionary(i, dataset.idx2char) for i in input[0]])
    print(dataset.idx2word[target[0]])
   
    while True:
        i, t, l = dataset.get_next_batch(batch_size=3000, mode='va')
        print(dataset.valid_ptr, len(i))
        if dataset.valid_ptr == 0:
            print('\niteration test pass!')
            break

