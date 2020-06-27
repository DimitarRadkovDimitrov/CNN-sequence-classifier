import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, models, layers, losses
from sentence_text_cnn import SentenceTextCNN
from sst_data_preprocessor import SSTDataPreprocessor

if __name__ == "__main__":
    num_classes = 2
    seq_length = 25
    embedding_dimensions = 300
    filter_sizes = [3, 4, 5]
    num_filters_per_size = 100
    dropout_rate = 0.5
    batch_size = 50

    cnn = SentenceTextCNN(
        num_classes=num_classes,
        seq_length=seq_length,
        embedding_dimensions=embedding_dimensions,
        filter_sizes=filter_sizes,
        num_filters_per_size=num_filters_per_size,
        dropout_rate=dropout_rate,
        batch_size=batch_size
    )
    preprocessor = SSTDataPreprocessor('./config.json', seq_length)
    preprocessor.print_analysis()
    train_x, train_y = preprocessor.get_training_embedded()
    dev_x, dev_y = preprocessor.get_dev_embedded()
    test_x, test_y = preprocessor.get_test_embedded()

    cnn.model.summary()
    cnn.train(train_x, train_y, dev_x, dev_y)
    report = cnn.evaluate(test_x, test_y)
    print(report)
