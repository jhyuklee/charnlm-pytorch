import numpy as np
import sys

class Dataset(object):

    def __init__(self, config):
        self.config = config
        self.total_data = self.process_data()

    def process_data(self):
        total_data = []
        with open(self.config.dataset_path) as f:
            for k, line in enumerate(f):
                csv_data = line[:-1].split(',')

        return total_data




if __name__ == '__main__':
    config = type('config', (object,), 
            {'dataset_path': './data/x0pr_20170903_174111.csv'})
    dataset = Dataset(config)

