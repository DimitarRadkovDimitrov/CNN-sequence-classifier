import csv
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sentence_text_cnn import SentenceTextCNN
from sst_data_preprocessor import SSTDataPreprocessor

if __name__ == "__main__":
    num_classes = 2
    seq_length = 40
    embedding_dimensions = 300
    filter_sizes = [3, 4, 5]
    num_filters_per_size = 100
    dropout_rate = 0.5
    batch_size = 50
    lr_decay = 0.95
    norm_lim = 3

    cnn = SentenceTextCNN(
        num_classes=num_classes,
        seq_length=seq_length,
        embedding_dimensions=embedding_dimensions,
        filter_sizes=filter_sizes,
        num_filters_per_size=num_filters_per_size,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        lr_decay=lr_decay,
        norm_lim=norm_lim
    )
    
    dataset = pickle.load(open('mr.p', 'rb'))
    train_x = dataset['train'][0]
    train_y = dataset['train'][1]
    dev_x = dataset['dev'][0]
    dev_y = dataset['dev'][1]
    test_x = dataset['test'][0]
    test_y = dataset['test'][1]

    cnn.model.summary()
    history = cnn.train(train_x, train_y, dev_x, dev_y, num_epochs=100)
    report = cnn.evaluate(test_x, test_y)
    print(report)

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
