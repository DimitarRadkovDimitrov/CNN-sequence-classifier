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


class SSTDataPreprocessor:
    """ Data preprocessing class """

    def __init__(self, config_filename):
        self.load_config_file(config_filename)
        self.num_classes = self.config['data']['num_classes']
        self.sequence_length = self.config['data']['seq_length']
        self.embedding_dimensions = self.config['data']['embedding_dims']
        self.static = self.config['data']['static']

        self.load_dataset()
        
        if self.static is True:
            if not os.path.isdir(self.config['models']['root']):
               os.mkdir(self.config['models']['root'])
               self.download_and_save_models()
            
            self.load_model_keyed_vectors()
            self.print_analysis()
            self.train = self.get_training_embedded()
            self.dev = self.get_dev_embedded()
            self.test = self.get_test_embedded()
        
        dataset = {}
        dataset['train'] = self.train
        dataset['dev'] = self.dev
        dataset['test'] = self.test
        self.save_data(dataset)


    def load_config_file(self, filename):
        with open(filename, "r") as f:
            self.config = json.load(f)


    def load_dataset(self):
        dataset = pytreebank.load_sst()

        if self.num_classes == 5:
            self.get_train_dev_test_fine(dataset)
        else:
            self.get_train_dev_test_course(dataset)


    def download_and_save_models(self):
        w2v_model = self.download_w2v_pretrained()
        w2v_model.wv.save_word2vec_format(self.config['models']['w2v_static'], binary=True)
        #w2v_model = self.train_w2v_sentiment_analysis(w2v_model)
        #w2v_model.wv.save_word2vec_format(self.config['models']['w2v_non_static'], binary=True)


    def download_w2v_pretrained(self):
        print("DOWNLOADING WORD2VEC PRETRAINED MODEL...")
        start_time = time.time()
        word2VecModel = downloader.load("word2vec-google-news-300")
        print("DONE. Time elapsed: {} seconds.".format(str(time.time() - start_time)))
        return word2VecModel

    
    def train_w2v_sentiment_analysis(self, model):
        #FIXME
        all_sentences = list(self.train) + list(self.dev) + list(self.test)
        model.build_vocab(all_sentences, update=True)
        model.train(all_sentences, total_examples=model.corpus_count, epochs=model.iter)
        return model


    def load_model_keyed_vectors(self):
        model_vector_map_name = ""

        if self.static is False:
            model_vector_map_name = self.config['models']['w2v_non_static']
        else:
            model_vector_map_name = self.config['models']['w2v_static']

        print("LOADING WORD2VEC PRETRAINED KEYED VECTORS INTO MEMORY...")
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

            all_sentences = list(self.train[0]) + list(self.dev[0]) + list(self.test[0])
            total_num_sentences = len(all_sentences)
            vocab = self.get_vocab(all_sentences)
            word2VecVocab = self.get_w2v_vocab(vocab)
            min_sentence_len, avg_sentence_len, max_sentence_len = self.get_sentence_stats(all_sentences)
            label_frequencies = self.get_label_stats(self.train[1])

            row = [
                'SST-{}'.format(self.num_classes), 
                self.num_classes,
                total_num_sentences,
                len(vocab),
                len(word2VecVocab),
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
                if word not in vocab:
                    vocab.add(word)
        
        return vocab
    

    def get_w2v_vocab(self, vocab):
        w2v_vocab = set()

        for word in vocab:
            if word in self.word2VecModel and word not in w2v_vocab:
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


    def get_training_embedded(self):
        sentences = self.train[0]
        labels = np.array(self.train[1])
        sentences = np.array(self.generate_sentence_embeddings(sentences))
        return [sentences, labels]


    def get_dev_embedded(self):
        sentences = self.dev[0]
        labels = np.array(self.dev[1])
        sentences = np.array(self.generate_sentence_embeddings(sentences))
        return [sentences, labels]


    def get_test_embedded(self):
        sentences = self.test[0]
        labels = np.array(self.test[1])
        sentences = np.array(self.generate_sentence_embeddings(sentences))
        return [sentences, labels]


    def generate_sentence_embeddings(self, sentences):
        sentence_embeddings = []

        for sentence in sentences:
            word_embeddings = self.generate_word_embeddings_from_sentence(sentence)
            sentence_embeddings.append(word_embeddings)

        return sentence_embeddings


    def generate_word_embeddings_from_sentence(self, sentence):
        counter = 0
        word_embeddings = []

        for word in sentence.split(" "):
            if counter == self.sequence_length:
                break
            try:
                word_embeddings.append(self.word2VecModel[word])
            except KeyError as error:
                random_word_embedding = np.random.default_rng().uniform(-0.25, 0.25, self.embedding_dimensions)
                word_embeddings.append(random_word_embedding)
            counter += 1

        for i in range(counter, self.sequence_length):
            word_embeddings.append([0] * 300)
    
        return word_embeddings


    def save_data(self, dataset):
        pickle.dump(dataset, open(self.config['data']['output'], 'wb'))
        print('dataset dictionary w/ \'train\', \'dev\', \'test\' keys as {}'.format(
            self.config['data']['output']
        ))


if __name__ == "__main__":
    preprocessor = SSTDataPreprocessor('./config.json')
