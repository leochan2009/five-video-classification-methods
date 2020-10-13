"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import coral_ordinal as coral
import pandas as pd
import time
import os.path
import numpy as np
import matplotlib.pyplot as plt

def predict(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, features_length = 14):

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit, modelName = 'coral_ordinal'
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape, modelName = 'coral_ordinal'
        )


    X, y = data.get_all_sequences_in_memory('train', data_type)
      # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model, features_length = features_length)

    ordinal_logits = rm.model.predict(X)
    # Convert from logits to label probabilities. This is initially a tensorflow tensor.
    tensor_probs = coral.ordinal_softmax(ordinal_logits)

    # Convert the tensor into a pandas dataframe.
    probs_df = pd.DataFrame(tensor_probs.numpy())

    probs_df.head()
    labels = probs_df.idxmax(axis=1)
    plt.plot(labels, label = 'current modeled')
    plt.plot(y, label='accessed')
    plt.show()
    np.save('predicted.npy',labels.values)
    np.save('y.npy', y)
    return labels.values


def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    modeled_pre = np.load('predicted-006-0.625.npy')
    #modeled = np.load('predicted.npy')
    accessed = np.load('y.npy')
    plt.plot(modeled_pre, label = "modeled")
    #plt.plot(modeled, label="current modeled")
    plt.plot(accessed+0.05, label = 'accessed')
    plt.legend()
    plt.show()

    model = 'coral_ordinal_lrcn'
    saved_model = 'data/checkpoints/coral_ordinal_lrcn-images.020-0.590.hdf5' #"data/checkpoints/lstm-features.456-0.148.hdf5" # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 30
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 8
    if model in ['simple', 'coral_ordinal']:
        data_type = 'features'
        image_shape = None
    elif model in ['coral_ordinal_lrcn']:
        data_type = 'images'
        image_shape = (250, 250, 3)
    else:
        raise ValueError("Invalid model. See train.py for options.")

    prediction = predict(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size)

    print(prediction)

if __name__ == '__main__':
    main()
