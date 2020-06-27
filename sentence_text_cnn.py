import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, layers, Model, backend, losses, callbacks
from sklearn.metrics import classification_report

class SentenceTextCNN:
    """ Text Classification Convolutional Neural Network """

    def __init__(self, **kwargs):
        valid_keys = [
            'seq_length', 
            'embedding_dimensions',
            'filter_sizes',
            'num_filters_per_size', 
            'num_classes',
            'dropout_rate',
            'batch_size'
        ]
        
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))

        self.model = self.build_model()

    
    def build_model(self):
        input_shape = (self.seq_length, self.embedding_dimensions, 1)
        inputs = Input(shape=input_shape)
    
        concatenate = layers.Concatenate(axis=-1)
        flatten = layers.Flatten(data_format='channels_last')
        dropout = layers.Dropout(self.dropout_rate)
        dense = layers.Dense(self.num_classes)
        softmax = layers.Softmax(axis=-1)
        
        pooled_outputs = []

        for filter_size in self.filter_sizes:
            filter_shape = (filter_size, self.embedding_dimensions)

            conv2d = layers.Conv2D(
                self.num_filters_per_size,
                filter_shape,
                1,
                padding='VALID',
                activation='relu',
                use_bias=True,
                data_format='channels_last',
                input_shape=input_shape
            )

            x = conv2d(inputs)
            x = backend.max(x, axis=-3)
            x = flatten(x)
            pooled_outputs.append(x)
        
        pooled_outputs = concatenate(pooled_outputs)
        pooled_outputs = dropout(pooled_outputs, training=True)
        pooled_outputs = dense(pooled_outputs)
        pooled_outputs = softmax(pooled_outputs)
        
        model = Model(inputs=inputs, outputs=pooled_outputs, name='sentence_cnn')
        model.compile(
            loss=losses.binary_crossentropy,
            optimizer='adam',
            metrics=['accuracy']
        )
        return model


    def train(self, train_x, train_y, dev_x, dev_y, num_epochs=5):
        model_checkpoint = callbacks.ModelCheckpoint(
            './weights.hdf5',
            verbose=1,
            save_best_only=True
        )
        self.model.fit(
            train_x,
            train_y,
            self.batch_size,
            num_epochs,
            validation_data=(dev_x, dev_y),
            callbacks=[model_checkpoint]
        )


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
            target_names=target_names
        )
        return report
