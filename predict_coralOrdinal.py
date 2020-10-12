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
    labels.values


def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'coral_ordinal'
    saved_model = 'data/checkpoints/coral_ordinal-features.437-1.214.hdf5' #"data/checkpoints/lstm-features.456-0.148.hdf5" # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 30
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 8
    if model in ['simple', 'coral_ordinal']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    predict(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size)

if __name__ == '__main__':
    main()
