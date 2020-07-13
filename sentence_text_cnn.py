import pickle
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Model, layers, initializers, optimizers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report


class SentenceTextCNN:
    """ Text Classification Convolutional Neural Network """

    def __init__(self, config_filename):
        self.load_config_file(config_filename)
        self.num_classes = self.config['data']['num_classes']
        self.seq_length = self.config['data']['seq_length']
        self.embedding_dimensions = self.config['data']['embedding_dims']
        self.embedding_training = not(self.config['CNN']['static'])
        self.filter_sizes = self.config['CNN']['filter_sizes']
        self.num_filters_per_size = self.config['CNN']['output_filters_per_size']
        self.activation_fn = self.config['CNN']['activation_function']
        self.dropout_rate = self.config['CNN']['dropout_rate']
        self.batch_size = self.config['CNN']['batch_size']
        self.lr_decay = self.config['CNN']['lr_decay']
        self.num_epochs = self.config['CNN']['num_epochs']
        self.data = pickle.load(open(self.config['data']['output'], 'rb'))
        self.index_to_vector_map = self.data['index_to_vector_map']
        self.model = self.build_model()


    def load_config_file(self, filename):
        with open(filename, "r") as f:
            self.config = json.load(f)
    

    def build_model(self):
        input_shape = (self.seq_length,)
        inputs = Input(shape=input_shape, dtype='int32')
    
        embedding = layers.Embedding(
            len(self.index_to_vector_map),
            self.embedding_dimensions,
            input_length=self.seq_length,
            trainable=self.embedding_training
        )
        embedding.build((None, len(self.index_to_vector_map)))
        embedding.set_weights([self.index_to_vector_map])
        concatenate = layers.Concatenate(axis=-1)
        flatten = layers.Flatten(data_format='channels_last')
        dropout = layers.Dropout(self.dropout_rate)
        dense = layers.Dense(self.num_classes, activation='softmax')
        
        pooled_outputs = []
        x_embedding = embedding(inputs)
        x_embedding = tf.expand_dims(x_embedding, -1)

        for filter_size in self.filter_sizes:
            filter_shape = (filter_size, self.embedding_dimensions)
            
            conv2d = layers.Conv2D(
                self.num_filters_per_size,
                filter_shape,
                1,
                padding='VALID',
                activation=self.activation_fn,
                kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),
                bias_initializer=initializers.zeros(),
                use_bias=True,
                data_format='channels_last',
                input_shape=(self.seq_length, self.embedding_dimensions, 1)
            )
            x = conv2d(x_embedding)
            x = tf.nn.max_pool(
                x,
                ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID'
            )
            x = flatten(x)
            pooled_outputs.append(x)
        
        pooled_outputs = concatenate(pooled_outputs)
        pooled_outputs = dropout(pooled_outputs)
        pooled_outputs = dense(pooled_outputs)

        model = Model(inputs=inputs, outputs=pooled_outputs, name='sentence_cnn')
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.Adadelta(
                learning_rate=0.01,
                rho=self.lr_decay,
                epsilon=1e-6,
            ),
            metrics=['accuracy']
        )
        return model


    def train(self, train_x, train_y, dev_x=None, dev_y=None):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)      
        train_y = to_categorical(train_y, num_classes=self.num_classes)

        if dev_x is not None and dev_y is not None:
            dev_y = to_categorical(dev_y, num_classes=self.num_classes)
            history = self.model.fit(
                train_x,
                train_y,
                self.batch_size,
                self.num_epochs,
                validation_data=(dev_x, dev_y),
                callbacks=[early_stopping],
                shuffle=True
            )
        else:
            history = self.model.fit(
                train_x,
                train_y,
                self.batch_size,
                self.num_epochs,
                validation_split=0.20,
                callbacks=[early_stopping],
                shuffle=True
            )

        return history


    def evaluate(self, test_x, test_y):
        target_names = ['class {}'.format(i) for i in range(self.num_classes)]
 
        pred_y = self.model.predict(
            test_x,
            self.batch_size,
            verbose=1
        )
        pred_y = np.argmax(pred_y, axis=1)

        report = classification_report(
            test_y,
            pred_y,
            target_names=target_names,
            digits=4
        )
        return report
