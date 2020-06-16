import re
import os
import pytreebank
import pandas as pd
import numpy as np
import time
from gensim.models import KeyedVectors, Word2Vec

class DataPreprocessor:
    """ Data preprocessing class """

    def __init__(self, sequence_length, stopwords_path='./stopwords.txt', number_of_classes=2, dataset_path='./samples', word2Vec_path='./w2v_pretrained'):
        self.number_of_classes = number_of_classes
        self.sequence_length = sequence_length
        self.embedding_dimensions = 300
        self.stopwords = self.load_stopwords(stopwords_path)

        start_time = time.time()
        self.word2VecModel = self.get_w2w_pretrained(word2Vec_path)
        print(str(time.time() - start_time) + " seconds.")
        
        self.train, self.dev, self.test = self.load_datasets(dataset_path)


    def get_w2w_pretrained(self, filename):
        model = KeyedVectors.load_word2vec_format(filename, binary=True)
        return model

    
    def load_stopwords(self, stopwords_path):
        stopwords = set()

        if os.path.isfile(stopwords_path):
            with open(stopwords_path, "r") as input_file:
                for line in input_file.readlines():
                    stopwords.add(line[:-1])

        return stopwords


    def load_datasets(self, dataset_path):
        if not os.path.isdir(dataset_path):
            self.fprintf_datasets(dataset_path)

        if self.number_of_classes == 5:
            train = pd.read_csv('{}/sst-5_train'.format(dataset_path), sep='\t', header=None)
            dev = pd.read_csv('{}/sst-5_dev'.format(dataset_path), sep='\t', header=None)
            test = pd.read_csv('{}/sst-5_test'.format(dataset_path), sep='\t', header=None)
            return train, dev, test
        else:
            train = pd.read_csv('{}/sst-2_train'.format(dataset_path), sep='\t', header=None)
            dev = pd.read_csv('{}/sst-2_dev'.format(dataset_path), sep='\t', header=None)
            test = pd.read_csv('{}/sst-2_test'.format(dataset_path), sep='\t', header=None)
            return train, dev, test


    def fprintf_datasets(self, dataset_path):
        os.mkdir(dataset_path)
        dataset = pytreebank.load_sst()
        self.fprintf_fine_grained_dataset(dataset, dataset_path)
        self.fprintf_course_grained_dataset(dataset, dataset_path)


    def fprintf_fine_grained_dataset(self, dataset, dataset_path):
        self.fprintf_fine_grained_labeled_data(dataset['train'], '{}/sst-5_train'.format(dataset_path))        
        self.fprintf_fine_grained_labeled_data(dataset['dev'], '{}/sst-5_dev'.format(dataset_path))
        self.fprintf_fine_grained_labeled_data(dataset['test'], '{}/sst-5_test'.format(dataset_path))


    def fprintf_course_grained_dataset(self, dataset, dataset_path):
        self.fprintf_course_grained_labeled_data(dataset['train'], '{}/sst-2_train'.format(dataset_path))
        self.fprintf_course_grained_labeled_data(dataset['dev'], '{}/sst-2_dev'.format(dataset_path))
        self.fprintf_course_grained_labeled_data(dataset['test'], '{}/sst-2_test'.format(dataset_path))


    def fprintf_fine_grained_labeled_data(self, training_data, filename):
        with open(filename, "w") as output_file:
            for data in training_data:
                label, sentence = data.to_labeled_lines()[0]
                output_file.write("{}\t{}\n".format(sentence, label))


    def fprintf_course_grained_labeled_data(self, training_data, filename):
        with open(filename, "w") as output_file:
            for data in training_data:
                label, sentence = data.to_labeled_lines()[0]
                if label < 2:
                    output_file.write("{}\t{}\n".format(sentence, 0))
                elif label > 2:
                    output_file.write("{}\t{}\n".format(sentence, 1))

    
    def print_analysis(self):
        train_set_size = len(self.train)
        dev_set_size = len(self.dev)
        test_set_size = len(self.test)

        if train_set_size == 0 or dev_set_size == 0 or test_set_size == 0:
            print("Error: Unable to load datasets")
        else:
            total_data_size = train_set_size + dev_set_size + test_set_size
            print("Total dataset size: {}".format(total_data_size))
            print("Training: {} ({}%)".format(train_set_size, round(train_set_size / total_data_size * 100)))
            print("Development: {} ({}%)".format(dev_set_size, round(dev_set_size / total_data_size * 100)))
            print("Test: {} ({}%)".format(test_set_size, round(test_set_size / total_data_size * 100)))
            
            min_sentence_len, avg_sentence_len, max_sentence_len = self.get_sentence_stats(self.train[0])
            label_frequencies = self.get_label_stats(self.train[1])
            print("\nTotal number of sentences (training): {}".format(train_set_size))
            print("Minimum sentence length: {}".format(min_sentence_len))
            print("Average sentence length: {}".format(avg_sentence_len))
            print("Maximum sentence length: {}".format(max_sentence_len))

            if self.number_of_classes == 5:
                print("\nVery negative: {}".format(label_frequencies[0]))
                print("Negative: {}".format(label_frequencies[1]))
                print("Neutral: {}".format(label_frequencies[2]))
                print("Positive: {}".format(label_frequencies[3]))
                print("Very positive: {}".format(label_frequencies[4]))
            else:
                print("\nNegative: {}".format(label_frequencies[0]))
                print("Positive: {}".format(label_frequencies[1]))


    def get_sentence_stats(self, sentences):
        number_of_sentences = len(sentences)

        if number_of_sentences == 0:
            return 0, 0, 0

        min_sentence_len = len(sentences[0].split(" "))
        avg_sentence_len = min_sentence_len
        max_sentence_len = min_sentence_len

        for i in range(1, number_of_sentences):
            sentence_length = len(sentences[i].split(" "))
            min_sentence_len = min((min_sentence_len, sentence_length))
            max_sentence_len = max((max_sentence_len, sentence_length))
            avg_sentence_len += sentence_length

        avg_sentence_len = round(avg_sentence_len / number_of_sentences)
        return min_sentence_len, avg_sentence_len, max_sentence_len


    def get_label_stats(self, labels):
        label_frequencies = {}

        if self.number_of_classes == 5:
            label_frequencies[0] = np.count_nonzero(labels == 0)
            label_frequencies[1] = np.count_nonzero(labels == 1)
            label_frequencies[2] = np.count_nonzero(labels == 2)
            label_frequencies[3] = np.count_nonzero(labels == 3)
            label_frequencies[4] = np.count_nonzero(labels == 4)
        else:
            label_frequencies[0] = np.count_nonzero(labels == 0)
            label_frequencies[1] = np.count_nonzero(labels == 1)

        return label_frequencies


    def get_training_embedded(self):
        sentences = self.train[0]
        labels = self.train[1]
        sentences = self.generate_sentence_embeddings(sentences)
        return [sentences, labels]


    def get_dev_embedded(self):
        sentences = self.dev[0]
        labels = self.dev[1]
        sentences = self.generate_sentence_embeddings(sentences)
        return [sentences, labels]


    def get_test_embedded(self):
        sentences = self.test[0]
        labels = self.test[1]
        sentences = self.generate_sentence_embeddings(sentences)
        return [sentences, labels]


    def generate_sentence_embeddings(self, sentences):
        sentence_embeddings = []

        for sentence in sentences:
            word_embeddings = self.generate_word_embeddings_with_padding(sentence)
            sentence_embeddings.append(word_embeddings)

        return sentence_embeddings


    def generate_word_embeddings_with_padding(self, sentence):
        counter = 0
        word_embeddings = []

        for word in sentence.split(" "):
            if counter == self.sequence_length:
                break
            try:
                word_embeddings.append(self.word2VecModel[word])
            except KeyError as error:
                random_word_embedding = np.random.default_rng().uniform(-0.99999, 1.0, self.embedding_dimensions)
                word_embeddings.append(random_word_embedding)
            counter += 1

        if len(word_embeddings) < self.sequence_length:
            length_to_extend = self.sequence_length - len(word_embeddings)
            padding = [[0] * self.embedding_dimensions] * length_to_extend
            word_embeddings.extend(padding)

        return word_embeddings
