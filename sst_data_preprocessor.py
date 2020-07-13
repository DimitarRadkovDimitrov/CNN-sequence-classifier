import re
import json
import os
import pytreebank
import numpy as np
import time
import pickle
import sys
from prettytable import PrettyTable
from gensim import downloader
from gensim.models import KeyedVectors, Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class SSTDataPreprocessor:
    """ Data preprocessing class """

    def __init__(self, config_filename):
        self.load_config_file(config_filename)
        self.num_classes = self.config['data']['num_classes']
        self.sequence_length = self.config['data']['seq_length']
        self.embedding_dimensions = self.config['data']['embedding_dims']

        self.load_data()
        self.fit_tokenizer()
        
        if not os.path.isdir(self.config['models']['root']):
            os.mkdir(self.config['models']['root'])
            self.download_and_save_model()
        
        self.load_word2vec_model()
        self.print_analysis()
        index_to_vector_map = self.get_index_to_vector_map()
        self.index_dataset()

        data = {}
        data['train'] = self.train
        data['dev'] = self.dev
        data['test'] = self.test
        data['index_to_vector_map'] = index_to_vector_map 
        self.save_data(data)


    def load_config_file(self, filename):
        with open(filename, "r") as f:
            self.config = json.load(f)


    def load_data(self):
        self.load_dataset()
        self.all_sentences = list(self.train[0]) + list(self.dev[0]) + list(self.test[0])
        self.vocab = self.get_vocab(self.all_sentences)


    def load_dataset(self):
        dataset = pytreebank.load_sst()

        if self.num_classes == 5:
            self.get_train_dev_test_fine(dataset)
        else:
            self.get_train_dev_test_course(dataset)


    def fit_tokenizer(self):
        self.tokenizer = Tokenizer(num_words=len(self.vocab))
        self.tokenizer.fit_on_texts(self.all_sentences)


    def download_and_save_model(self):
        w2v_model = self.download_w2v_pretrained()
        w2v_model.wv.save_word2vec_format(self.config['models']['w2v_static'], binary=True)


    def download_w2v_pretrained(self):
        print("DOWNLOADING WORD2VEC GOOGLE NEWS PRETRAINED MODEL...")
        start_time = time.time()
        word2VecModel = downloader.load("word2vec-google-news-300")
        print("DONE. Time elapsed: {} seconds.".format(str(time.time() - start_time)))
        return word2VecModel


    def load_word2vec_model(self):
        self.load_model_keyed_vectors()
        self.w2v_vocab = self.get_w2v_vocab(self.vocab)    


    def load_model_keyed_vectors(self):
        model_vector_map_name = self.config['models']['w2v_static']
        print("LOADING STATIC WORD2VEC KEYED VECTORS INTO MEMORY...")
        start_time = time.time()
        self.word2VecModel = KeyedVectors.load_word2vec_format(model_vector_map_name, binary=True)
        print("DONE. Time elapsed: {} seconds.".format(str(time.time() - start_time)))


    def get_train_dev_test_fine(self, dataset):
        self.train = self.get_fine_grained_labeled_data(dataset['train'])
        self.dev = self.get_fine_grained_labeled_data(dataset['dev'])
        self.test = self.get_fine_grained_labeled_data(dataset['test'])


    def get_train_dev_test_course(self, dataset):
        self.train = self.get_course_grained_labeled_data(dataset['train'])
        self.dev = self.get_course_grained_labeled_data(dataset['dev'])
        self.test = self.get_course_grained_labeled_data(dataset['test'])


    def get_fine_grained_labeled_data(self, training_data):
        data_x = []
        data_y = []

        for data in training_data:
            label, sentence = data.to_labeled_lines()[0]
            sentence = self.clean_sentence(sentence)
            data_x.append(sentence)
            data_y.append(int(label))

        return [data_x, data_y]


    def get_course_grained_labeled_data(self, training_data):
        data_x = []
        data_y = []

        for data in training_data:
            label, sentence = data.to_labeled_lines()[0]
            sentence = self.clean_sentence(sentence)
            if label < 2:
                data_x.append(sentence)
                data_y.append(0)
            elif label > 2:
                data_x.append(sentence)
                data_y.append(1)

        return [data_x, data_y]


    def clean_sentence(self, sentence):
        rev = [sentence.strip()]
        rev = self.clean_str_sst(' '.join(rev))
        return rev
    

    def clean_str_sst(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
        string = re.sub(r"\s{2,}", " ", string)    
        return string.strip().lower()
    

    def print_analysis(self):
        if self.train is None or self.dev is None or self.test is None:
            print("Error: Unable to load datasets")
        else:
            table = PrettyTable()
            table.field_names = ['Dataset', 'Number of Classes', 'N', '|V|', '|V - pre|', 'Test']

            total_num_sentences = len(self.all_sentences)
            min_sentence_len, avg_sentence_len, max_sentence_len = self.get_sentence_stats(self.all_sentences)
            label_frequencies = self.get_label_stats(self.train[1])

            row = [
                'SST-{}'.format(self.num_classes), 
                self.num_classes,
                total_num_sentences,
                len(self.vocab),
                len(self.w2v_vocab),
                len(self.test[0])
            ]
            table.add_row(row)
            print(table)
            
            table = PrettyTable()
            field_names = ['Total Sentences', 'Min', 'Avg', 'Max']
            row = [total_num_sentences, min_sentence_len, avg_sentence_len, max_sentence_len]

            if self.num_classes == 5:
                field_names.extend(['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
                row.extend([
                    label_frequencies[0], 
                    label_frequencies[1], 
                    label_frequencies[2], 
                    label_frequencies[3], 
                    label_frequencies[4]
                ])
            else:
                field_names.extend(['Negative', 'Positive'])
                row.extend([label_frequencies[0], label_frequencies[1]])

            table.field_names = field_names
            table.add_row(row)
            print(table)


    def get_vocab(self, all_sentences):
        vocab = set()
        
        for sentence in all_sentences:
            for word in sentence.split(' '):
                vocab.add(word)
        
        return vocab
    

    def get_w2v_vocab(self, vocab):
        w2v_vocab = set()

        for word in vocab:
            if word in self.word2VecModel:
                w2v_vocab.add(word)

        return w2v_vocab


    def get_sentence_stats(self, sentences):
        sentence_counts = [len(sentence.split(' ')) for sentence in sentences]
        min_sentence_len = np.amin(sentence_counts)
        max_sentence_len = np.amax(sentence_counts)
        avg_sentence_len = round(np.average(sentence_counts))
        return min_sentence_len, avg_sentence_len, max_sentence_len


    def get_label_stats(self, labels):
        label_frequencies = {}

        if self.num_classes == 5:
            label_frequencies[0] = labels.count(0)
            label_frequencies[1] = labels.count(1)
            label_frequencies[2] = labels.count(2)
            label_frequencies[3] = labels.count(3)
            label_frequencies[4] = labels.count(4)
        else:
            label_frequencies[0] = labels.count(0)
            label_frequencies[1] = labels.count(1)

        return label_frequencies


    def get_index_to_vector_map(self):
        index_to_vector_map = np.zeros((len(self.tokenizer.word_index) + 1, self.embedding_dimensions))
        for word, index in self.tokenizer.word_index.items():
            word_embedding = None
            try:
                word_embedding = self.word2VecModel[word]
            except KeyError as error:
                word_embedding = np.random.default_rng().uniform(-0.25, 0.25, self.embedding_dimensions)
            index_to_vector_map[index] = word_embedding
        return index_to_vector_map


    def index_dataset(self):
        self.train[0] = self.tokenizer.texts_to_sequences(self.train[0])
        self.dev[0] = self.tokenizer.texts_to_sequences(self.dev[0])
        self.test[0] = self.tokenizer.texts_to_sequences(self.test[0])
        self.train[0] = pad_sequences(self.train[0], maxlen=self.sequence_length, padding='post')
        self.dev[0] = pad_sequences(self.dev[0], maxlen=self.sequence_length, padding='post')
        self.test[0] = pad_sequences(self.test[0], maxlen=self.sequence_length, padding='post')
        self.train[1] = np.array(self.train[1])
        self.dev[1] = np.array(self.dev[1])
        self.test[1] = np.array(self.test[1]) 


    def save_data(self, dataset):
        pickle.dump(dataset, open(self.config['data']['output'], 'wb'))
        print('data dictionary w/ \'train\', \'dev\', \'test\', and \'index_to_vector_map\' keys as {}'.format(
            self.config['data']['output']
        ))


if __name__ == "__main__":
    path_to_config = sys.argv[1]
    preprocessor = SSTDataPreprocessor(path_to_config)
