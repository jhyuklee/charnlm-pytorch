import numpy as np
import sys

class Dataset(object):
    def __init__(self, config):
        self.config = config

        # dictionary settings
        self.char2idx = {}
        self.idx2char = {}
        self.word2idx = {}
        self.idx2word = {}
        self.UNK = 'UNK'
        self.PAD = 'PAD'
        self.initialize_dictionary()

        self.train_data = self.process_data(
                self.config.train_path, 
                update_dict=True)
        self.valid_data = self.process_data(
                self.config.valid_path)
        self.test_data = self.process_data(
                self.config.test_path)

        self.pad_data(self.train_data)
        self.pad_data(self.valid_data)
        self.pad_data(self.test_data)

        self.train_ptr = 0
        self.valid_ptr = 0
        self.test_ptr = 0
        
        print('char_dict', len(self.char2idx))
        print('word_dict', len(self.word2idx), end='\n\n')

    def initialize_dictionary(self):
        self.char2idx[self.UNK] = 0
        self.char2idx[self.PAD] = 1
        self.idx2char[0] = self.UNK
        self.idx2char[1] = self.PAD
        self.word2idx[self.UNK] = 0
        self.word2idx[self.PAD] = 1
        self.idx2word[0] = self.UNK
        self.idx2word[1] = self.PAD
    
    def update_dictionary(self, key, mode=None):
        if mode == 'c':
            if key not in self.char2idx:
                self.char2idx[key] = len(self.char2idx)
                self.idx2char[len(self.idx2char)] = key
        elif mode == 'w':
            if key not in self.word2idx:
                self.word2idx[key] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = key
    
    def map_dictionary(self, key_list, dictionary):
        output = []
        for key in key_list:
            if key in dictionary:
                output.append(dictionary[key])
            else: # no unknown found yet. (separate i2c, c2i)
                output.append(dictionary[self.UNK])
        return output

    def process_data(self, path, update_dict=False):
        print('processing %s' % path)
        total_data = []
        max_wordlen = max_sentlen = 0

        with open(path) as f:
            for k, line in enumerate(f):
                sentence = line[:-1]
                
                # count max length
                for word in sentence.split():
                    max_wordlen = (len(word)
                            if len(word) > max_wordlen else max_wordlen)
                max_sentlen = (len(sentence.split()) 
                        if len(sentence.split()) > max_sentlen else max_sentlen)

                if update_dict:
                    for char in sentence:
                        self.update_dictionary(char, 'c')
                    for word in sentence.split():
                        self.update_dictionary(word, 'w')
                    if max_wordlen > self.config.max_wordlen:
                        self.config.max_wordlen = max_wordlen
                    if max_sentlen > self.config.max_sentlen:
                        self.config.max_sentlen = max_sentlen
                
                sentchar = []
                for word in sentence.split():
                    sentchar.append(self.map_dictionary(word, self.char2idx))
                sentword = self.map_dictionary(sentence.split(), self.word2idx)
                total_data.append([sentchar, sentword])
                assert len(sentword) == len(sentchar)

        if update_dict:
            self.config.char_vocab_size = len(self.char2idx)
            self.config.word_vocab_size = len(self.word2idx)
            self.config.max_sentlen

        print('data size', len(total_data))
        print('max wordlen', max_wordlen)
        print('max sentlen', max_sentlen, end='\n\n')

        return total_data

    def pad_data(self, dataset):
        for data in dataset:
            sentchar, sentword = data
            # pad sentword (will be slided to right)
            while len(sentword) != self.config.max_sentlen + 1:
                sentword.append(self.word2idx[self.PAD])
            # pad word in sentchar
            for word in sentchar:
                while len(word) != self.config.max_wordlen:
                    word.append(self.char2idx[self.PAD])
            # pad sentchar
            while len(sentchar) != self.config.max_sentlen:
                sentchar.append([self.char2idx[self.PAD]] * self.config.max_wordlen)
            assert len(sentchar) == len(sentword) - 1
    
    def get_next_batch(self, mode='tr', batch_size=None, as_numpy=True):
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
        
        batch_size = (batch_size 
                if ptr + batch_size <= len(data)
                else len(data) - ptr)
        inputs = [d[0] for d in data[ptr:ptr+batch_size]]
        targets = [d[1][1:] for d in data[ptr:ptr+batch_size]]
        
        if mode == 'tr':
            self.train_ptr = (ptr + batch_size) % len(data)
        elif mode == 'va':
            self.valid_ptr = (ptr + batch_size) % len(data)
        elif mode == 'te':
            self.test_ptr = (ptr + batch_size) % len(data)

        if as_numpy:
            inputs = np.array(inputs)
            targets = np.array(targets)

        return inputs, targets
    
    def get_batch_ptr(self, mode):
        if mode == 'tr':
            return self.train_ptr
        elif mode == 'va':
            return self.valid_ptr
        elif mode == 'te':
            return self.test_ptr


if __name__ == '__main__':
    # Testing
    config = type('config', (object,), {
        'train_path': './data/train.txt',
        'valid_path': './data/valid.txt',
        'test_path': './data/test.txt',
        'batch_size': 32
    })
    dataset = Dataset(config)

    """
    input, target = dataset.get_next_batch(batch_size=1)
    print([dataset.map_dictionary(i, dataset.idx2char) for i in input[0]])
    print(dataset.map_dictionary(target[0], dataset.idx2word))
    """
    while True:
        i, t = dataset.get_next_batch(batch_size=300, mode='va')
        print(dataset.valid_ptr, len(i))
        if dataset.valid_ptr == 0:
            break

