import re
import json
import os
import pytreebank
import pandas as pd
import numpy as np
import time
from prettytable import PrettyTable
from gensim import downloader
from gensim.models import KeyedVectors, Word2Vec


class SSTDataPreprocessor:
    """ Data preprocessing class """

    def __init__(self, config_filename, sequence_length, static=True, number_of_classes=2):
        self.number_of_classes = number_of_classes
        self.sequence_length = sequence_length
        self.embedding_dimensions = 300
        self.static = static

        self.load_config_file(config_filename)

        if not os.path.isdir(self.config['datasets']['root']):
            os.mkdir(self.config['datasets']['root'])
            self.download_and_save_datasets()

        self.load_datasets()
        
        if not os.path.isdir(self.config['models']['root']):
            os.mkdir(self.config['models']['root'])
            self.download_and_save_models()
        
        self.load_model_keyed_vectors()


    def load_config_file(self, filename):
        with open(filename, "r") as f:
            self.config = json.load(f)


    def download_and_save_datasets(self):
        dataset = pytreebank.load_sst()
        self.save_fine_grained_dataset(dataset, self.config['datasets']['sst_5'])
        self.save_course_grained_dataset(dataset, self.config['datasets']['sst_2'])


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


    def load_datasets(self):
        dataset_root_filename = None

        if self.number_of_classes == 5:
            dataset_root_filename = self.config['datasets']['sst_5']
        else:
            dataset_root_filename = self.config['datasets']['sst_2']

        self.train = pd.read_csv(dataset_root_filename['train'], sep='\t', header=None)
        self.dev = pd.read_csv(dataset_root_filename['dev'], sep='\t', header=None)
        self.test = pd.read_csv(dataset_root_filename['test'], sep='\t', header=None)


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


    def save_fine_grained_dataset(self, dataset, dataset_root_dir):
        self.fprintf_fine_grained_labeled_data(dataset['train'], dataset_root_dir['train'])        
        self.fprintf_fine_grained_labeled_data(dataset['dev'], dataset_root_dir['dev'])
        self.fprintf_fine_grained_labeled_data(dataset['test'], dataset_root_dir['test'])


    def save_course_grained_dataset(self, dataset, dataset_root_dir):
        self.fprintf_course_grained_labeled_data(dataset['train'], dataset_root_dir['train'])
        self.fprintf_course_grained_labeled_data(dataset['dev'], dataset_root_dir['dev'])
        self.fprintf_course_grained_labeled_data(dataset['test'], dataset_root_dir['test'])


    def fprintf_fine_grained_labeled_data(self, training_data, filename):
        with open(filename, "w") as output_file:
            for data in training_data:
                label, sentence = data.to_labeled_lines()[0]
                sentence = self.clean_sentence(sentence)
                output_file.write("{}\t{}\n".format(sentence, label))


    def fprintf_course_grained_labeled_data(self, training_data, filename):
        with open(filename, "w") as output_file:
            for data in training_data:
                label, sentence = data.to_labeled_lines()[0]
                sentence = self.clean_sentence(sentence)
                if label < 2:
                    output_file.write("{}\t{}\n".format(sentence, 0))
                elif label > 2:
                    output_file.write("{}\t{}\n".format(sentence, 1))


    def clean_sentence(self, sentence):
        new_sentence = ''

        for word in sentence.split(' '):
            new_word = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", word)   
            new_word = re.sub(r"\s{2,}", " ", new_word).strip().lower() 
            new_sentence += new_word + ' '

        return new_sentence
    
    
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
                'SST-{}'.format(self.number_of_classes), 
                self.number_of_classes,
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

            if self.number_of_classes == 5:
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
        labels = self.train[1].to_numpy()
        sentences = np.array(self.generate_sentence_embeddings(sentences))
        return [sentences, labels]


    def get_dev_embedded(self):
        sentences = self.dev[0]
        labels = self.dev[1].to_numpy()
        sentences = np.array(self.generate_sentence_embeddings(sentences))
        return [sentences, labels]


    def get_test_embedded(self):
        sentences = self.test[0]
        labels = self.test[1].to_numpy()
        sentences = np.array(self.generate_sentence_embeddings(sentences))
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
