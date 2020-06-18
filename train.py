import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, models, layers, losses
from sentence_text_cnn import SentenceTextCNN
from data_preprocessor import DataPreprocessor

if __name__ == "__main__":
    dummy_input_x = [[], []]
    dummy_input_x[0] = [np.random.default_rng().uniform(-0.99999, 1.0, 300)]
    dummy_input_x[0].extend([[0] * 300] * 19)
    dummy_input_x[1] = [np.random.default_rng().uniform(-0.99999, 1.0, 300)]
    dummy_input_x[1].extend([[0] * 300] * 19)
    dummy_input_y = [1, 0]

    dummy_val_x = [[], []]
    dummy_val_x[0] = [np.random.default_rng().uniform(-0.99999, 1.0, 300)]
    dummy_val_x[0].extend([[0] * 300] * 19)
    dummy_val_x[1] = [np.random.default_rng().uniform(-0.99999, 1.0, 300)]
    dummy_val_x[1].extend([[0] * 300] * 19)
    dummy_val_y = [0, 0]

    dummy_test_x = [[]]
    dummy_test_x[0] = [np.random.default_rng().uniform(-0.99999, 1.0, 300)]
    dummy_test_x[0].extend([[0] * 300] * 19)
    dummy_test_y = [1]

    dummy_input_x = np.array(dummy_input_x)
    dummy_input_y = np.array(dummy_input_y)
    dummy_val_x = np.array(dummy_val_x)
    dummy_val_y = np.array(dummy_val_y)
    dummy_test_x = np.array(dummy_test_x)
    dummy_test_y = np.array(dummy_test_y)
    
    dummy_input_x = dummy_input_x.reshape((2, 20, 300, 1))
    dummy_val_x = dummy_val_x.reshape((2, 20, 300, 1))
    dummy_test_x = dummy_test_x.reshape((1, 20, 300, 1))
    


    # train_x, train_y = preprocessor.get_training_embedded()
    # dev_x, dev_y = preprocessor.get_dev_embedded()

    cnn = SentenceTextCNN(
        num_classes=2,
        seq_length=20,
        embedding_dimensions=300,
        filter_sizes=[3, 4, 5],
        num_filters_per_size=100,
        dropout_rate=0.5
    )
    cnn.train(dummy_input_x, dummy_input_y, dummy_val_x, dummy_val_y)
    predictions = cnn.predict(dummy_test_x)
    print(predictions)
