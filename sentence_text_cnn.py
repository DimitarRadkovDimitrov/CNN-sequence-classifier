import tensorflow as tf
from tensorflow.keras import Input, layers, Model, backend, losses

class SentenceTextCNN:
    """ Text Classification Convolutional Neural Network """

    def __init__(self, **kwargs):
        valid_keys = [
            'seq_length', 
            'embedding_dimensions',
            'filter_sizes',
            'num_filters_per_size', 
            'num_classes',
            'dropout_rate'
        ]
        
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))

        input_shape = (self.seq_length, self.embedding_dimensions, 1)
        self.model = self.build_model()

    
    def build_model(self):
        input_shape = (self.seq_length, self.embedding_dimensions, 1)
        inputs = Input(shape=input_shape)
    
        concatenate = layers.Concatenate(axis=-1)
        flatten = layers.Flatten(data_format='channels_last')
        dropout = layers.Dropout(self.dropout_rate)
        softmax = layers.Softmax(axis=-1)
        pooled_outputs = []

        for i, filter_size in enumerate(self.filter_sizes):
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
        pooled_outputs = dropout(pooled_outputs)
        pooled_outputs = softmax(pooled_outputs)
        
        model = Model(inputs=inputs, outputs=pooled_outputs, name='sentence_cnn')
        model.summary()
        model.compile(
            loss=losses.binary_crossentropy,
            optimizer='adam'
        )
        return model


    def train(self, x_train, y_train, x_dev, y_dev):
        self.model.fit(
            x_train,
            y_train,
            validation_data=(x_dev, y_dev)
        )


    def predict(self, x_test):
        predictions = self.model.predict(x_test)
        return predictions
