import re
import json
import os
import pytreebank
import numpy as np
import time
import pickle
import sys
import torch
import transformers
from prettytable import PrettyTable
from gensim import downloader
from gensim.models import KeyedVectors, Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class SSTDataPreprocessor:
    """ Data preprocessing class """

    def __init__(self, config):
        self.config = config
        self.num_classes = self.config['data']['num_classes']
        self.sequence_length = self.config['data']['seq_length']
        self.bert_embeddings = self.config['data']['bert_embeddings']
        self.embedding_dimensions = self.config['data']['embedding_dims']

        self.load_data()

        if not os.path.isdir(self.config['models']['root']):
            os.mkdir(self.config['models']['root'])
            self.download_and_save_w2v_model()

        if self.bert_embeddings is True:
            self.download_bert_model_and_tokenizer()
            self.generate_bert_embeddings()
            index_to_vector_map = self.get_index_to_vector_map_bert()
        else:
            self.fit_tokenizer()
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


    def get_vocab(self, all_sentences):
        vocab = set()
        for sentence in all_sentences:
            for word in sentence.split(' '):
                vocab.add(word)
        return vocab


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


    def fit_tokenizer(self):
        self.tokenizer = Tokenizer(num_words=len(self.vocab))
        self.tokenizer.fit_on_texts(self.all_sentences)


    def download_and_save_w2v_model(self):
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


    def get_w2v_vocab(self, vocab):
        w2v_vocab = set()
        for word in vocab:
            if word in self.word2VecModel:
                w2v_vocab.add(word)
        return w2v_vocab


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


    def download_bert_model_and_tokenizer(self):
        print("DOWNLOADING BERT PRETRAINED MODEL INTO MEMORY...")
        start_time = time.time()
        model_class, tokenizer_class, pretrained_weights = (
            transformers.BertModel,
            transformers.BertTokenizer,
            'bert-base-uncased'
        )
        self.bert_model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
        self.bert_model.eval()
        self.bert_tokenizer = tokenizer_class.from_pretrained(
            pretrained_weights, 
            cache_dir=self.config['models']['root']
        )
        print("DONE. Time elapsed: {} seconds.".format(str(time.time() - start_time)))


    def generate_bert_embeddings(self):
        self.train[0] = self.get_sentences_embedded_bert(self.train[0])
        self.dev[0] = self.get_sentences_embedded_bert(self.dev[0])
        self.test[0] = self.get_sentences_embedded_bert(self.test[0])
        self.train[1] = np.array(self.train[1])
        self.dev[1] = np.array(self.dev[1])
        self.test[1] = np.array(self.test[1]) 


    def get_sentences_embedded_bert(self, sentences):
        all_sentences = self.encode_sentences_bert(sentences)
        segment_ids = [[1] * len(sentence) for sentence in all_sentences]
        input_ids = torch.tensor(all_sentences).to(torch.int64)
        segment_ids = torch.tensor(segment_ids)

        print("RUNNING BERT PRE-TRAINED MODEL...")

        start_time = time.time()
        with torch.no_grad():
            hidden_states = self.bert_model(input_ids, segment_ids)[2]

        print("DONE. Time elapsed: {} seconds.".format(str(time.time() - start_time)))
        
        sentence_embeddings = torch.stack(hidden_states, dim=0)
        sentence_embeddings = sentence_embeddings.permute(1, 2, 0, 3)

        final_sentence_vectors = []
        for sentence_embedding in sentence_embeddings:
            token_vec_sums = []

            for token_embedding in sentence_embedding:
                vector_sum = torch.sum(token_embedding[-4:], dim=0)
                token_vec_sums.append(vector_sum.numpy())

            final_sentence_vectors.append(np.array(token_vec_sums))

        return np.array(final_sentence_vectors)


    def encode_sentences_bert(self, sentences):
        sentences_encoded = []
        for sentence in sentences:
            sentence_encoded = np.array(self.bert_tokenizer.encode(sentence, add_special_tokens=True))
            sentences_encoded.append(sentence_encoded)            
        return pad_sequences(sentences_encoded, maxlen=self.sequence_length, padding='post')


    def get_index_to_vector_map_bert(self):
        index_to_vector_map = [[0] * self.embedding_dimensions]
        self.train[0] = self.index_dataset_bert(self.train[0], index_to_vector_map)
        self.dev[0] = self.index_dataset_bert(self.dev[0], index_to_vector_map)
        self.test[0] = self.index_dataset_bert(self.test[0], index_to_vector_map)       
        return np.array(index_to_vector_map)


    def index_dataset_bert(self, embedded_sentences, index_to_vector_map):
        array_of_indices = []
        map_index = len(index_to_vector_map)

        for sentence_index, embedded_sentence in enumerate(embedded_sentences):
            array_of_indices.append([])
            for word_index, word_embedding in enumerate(embedded_sentence):
                array_of_indices[sentence_index].append(map_index)
                map_index += 1
                index_to_vector_map.append(word_embedding)
            array_of_indices[sentence_index] = np.array(array_of_indices[sentence_index])
        return np.array(array_of_indices)


    def save_data(self, dataset):
        pickle.dump(dataset, open(self.config['data']['output'], 'wb'))
        print('data dictionary w/ \'train\', \'dev\', \'test\', and \'index_to_vector_map\' keys as {}'.format(
            self.config['data']['output']
        ))


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


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)
    preprocessor = SSTDataPreprocessor(config)
