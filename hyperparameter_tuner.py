import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import pickle
import itertools
import numpy as np
from sentence_text_cnn import SentenceTextCNN
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

class HyperparameterTuner:
    """ Hyperparameter tuning class """
    
    def __init__(self, train, dev, config, cross_val=True):
        self.baseline_config = config
        self.split = 5
        
        if cross_val == False:
            train_samples = [-1 for i in range(len(train[0]))]
            dev_samples = [0 for i in range(len(dev[0]))]
            self.split = PredefinedSplit(test_fold=np.concatenate((train_samples, dev_samples)))
        
        self.activation_function = self.baseline_config['CNN']['activation_function']
        self.filter_sizes = self.baseline_config['CNN']['filter_sizes']
        self.output_filters_per_size = self.baseline_config['CNN']['output_filters_per_size']
        self.dropout_rate = self.baseline_config['CNN']['dropout_rate']
        self.batch_size = None
        self.epochs = None

        self.train_x = np.concatenate((train[0], dev[0]))
        self.train_y = np.concatenate((train[1], dev[1])) 
        self.batch_size, self.epochs = self.best_batch_size_and_epochs()
        individual_filter_size = self.best_individual_filter_size()
        self.filter_sizes = self.best_filter_size_combination(individual_filter_size)
        best_results = self.best_model_configuration()
        self.output_filters_per_size = best_results['output_filters_per_size']
        self.activation_function = best_results['activation_function']
        self.dropout_rate = best_results['dropout_rate']

        print('Optimal configuration:')
        self.print_configuration()


    def best_batch_size_and_epochs(self):
        print('Finding optimal batch size, epochs combination with baseline configuration:')
        self.print_configuration()

        batch_size = [32, 64, 128, 256, 512, 1024]
        epochs = [10, 50, 100, 500, 1000]

        param_grid = dict(batch_size=batch_size, epochs=epochs)
        model = KerasClassifier(build_fn=self.build_fn, verbose=0)
        clf = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=self.split)
        grid_result = clf.fit(self.train_x, self.train_y)
        self.print_grid_result(grid_result)
        return grid_result.best_params_['batch_size'], grid_result.best_params_['epochs']


    def best_individual_filter_size(self):
        print('Finding optimal individual filter size with:')
        self.filter_sizes = None
        self.print_configuration()

        filter_sizes = list(zip(range(1, 11)))
        param_grid = dict(filter_sizes=filter_sizes)
        grid_result = self.get_grid_result(param_grid)
        self.print_grid_result(grid_result)
        return grid_result.best_params_['filter_sizes']


    def best_filter_size_combination(self, best_ind_size):
        print(f'Finding optimal filter size combination around {best_ind_size}:')
        self.filter_sizes = None
        self.print_configuration()

        filter_sizes = [best_ind_size - 1, best_ind_size, best_ind_size + 1]
        filter_heights_of_size_three = [tuple(i) for i in itertools.combinations_with_replacement(filter_sizes, 3)]
        filter_heights_of_size_four = [tuple(i) for i in itertools.combinations_with_replacement(filter_sizes, 4)]
        all_filter_combinations = filter_heights_of_size_three + filter_heights_of_size_four

        param_grid = dict(filter_sizes=all_filter_combinations)
        grid_result = self.get_grid_result(param_grid)
        self.print_grid_result(grid_result)
        return grid_result.best_params_['filter_sizes']


    def best_model_configuration(self):
        print('Finding optimal model configuration with:')
        self.print_configuration()

        output_filters_per_size = [100, 200, 300, 400, 500, 600]
        activation_function = ['relu', 'tanh']
        dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5]

        param_grid = dict(
            output_filters_per_size=output_filters_per_size,
            activation_function=activation_function,
            dropout_rate=dropout_rate
        )
        grid_result = self.get_grid_result(param_grid)
        self.print_grid_result(grid_result)
        return grid_result.best_params_


    def get_grid_result(self, param_grid):
        model = KerasClassifier(build_fn=self.build_fn, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        clf = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=self.split)
        grid_result = clf.fit(self.train_x, self.train_y)
        return grid_result


    def build_fn(self, filter_sizes=None, output_filters_per_size=None, activation_function=None, dropout_rate=None):
        config_copy = dict(self.baseline_config)

        for arg, value in locals().items():
            if value is None:
                config_copy['CNN'][arg] = getattr(self, arg)

        cnn = SentenceTextCNN(config_copy)
        return cnn.model


    def print_configuration(self):
        print(f'Activation function: {self.activation_function}')
        print(f'Filter sizes: {self.filter_sizes}')
        print(f'Number of feature maps per filter: {self.output_filters_per_size}')
        print(f'Dropout rate: {self.dropout_rate}')
        print(f'Number of epochs: {self.epochs}')
        print(f'Mini-batch size: {self.batch_size}\n')
    

    def print_grid_result(self, grid_result):
        print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        params = grid_result.cv_results_['params']

        for mean, param in zip(means, params):
            print('\t%f with: %r' % (mean, param))
        print()


def get_baseline_config():
    config = {}
    config['data'] = {}
    config['CNN'] = {}
    config['data']['num_classes'] = 2
    config['data']['seq_length'] = 53
    config['data']['embedding_dims'] = 300
    config['data']['output'] = './data.bin'
    config['CNN']['static'] = True
    config['CNN']['activation_function'] = 'relu'
    config['CNN']['filter_sizes'] = [3, 4, 5]
    config['CNN']['output_filters_per_size'] = 100
    config['CNN']['dropout_rate'] = 0.5
    config['CNN']['batch_size'] = 50
    config['CNN']['learning_rate'] = 0.01
    config['CNN']['lr_decay'] = 0.95
    config['CNN']['num_epochs'] = 50
    return config


if __name__ == '__main__':
    config = get_baseline_config()
    data = pickle.load(open(config['data']['output'], 'rb'))
    parameter_tuner = HyperparameterTuner(data['train'], data['dev'], config, cross_val=False)
