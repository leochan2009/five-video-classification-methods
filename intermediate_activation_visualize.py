"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
from keras.preprocessing import image
import coral_ordinal as coral
import pandas as pd
import time
import os.path
import numpy as np
import matplotlib.pyplot as plt
from keras import models

def predict(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, features_length = 14):

    # Get the data and process it.
    if image_shape is None:
        if model in ['coral_ordinal', 'coral_ordinal_lrcn']:
            data = DataSet(
                seq_length=seq_length,
                class_limit=class_limit, labelEncoding='coral_ordinal'
            )
        else:
            data = DataSet(
                seq_length=seq_length,
                class_limit=class_limit, labelEncoding='lstm'
            )
    else:
        if model in ['coral_ordinal', 'coral_ordinal_lrcn']:
            data = DataSet(
                seq_length=seq_length,
                class_limit=class_limit,
                image_shape=image_shape, labelEncoding='coral_ordinal'
            )
        else:
            data = DataSet(
                seq_length=seq_length,
                class_limit=class_limit,
                image_shape=image_shape, labelEncoding='lstm'
            )


    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model, features_length = features_length)
    X, y = data.get_all_sequences_in_memory('train', data_type)
    if model in ['coral_ordinal_lrcn','coral_ordinal']:
        layer_outputs = [layer.output for layer in rm.model.layers[:35]]
        # Extracts the outputs of the top 12 layers
        activation_model = models.Model(inputs=rm.model.input,
                                        outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input
        img_tensor = X[0]
        img_tensor = np.reshape(img_tensor, (1, 30, 250, 250, 3))
        plt.imshow(img_tensor[0][1])
        plt.show()
        activations = activation_model.predict(img_tensor)
        first_layer_activation = activations[0]
        print(first_layer_activation.shape)
        plt.matshow(first_layer_activation[0, 0, :, :, 4], cmap='viridis')
        plt.show()
        layer_names = []
        for layer in rm.model.layers[:35]:
            layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

        images_per_row = 16
        for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
            n_features = layer_activation.shape[-1]  # Number of features in the feature map
            size = layer_activation.shape[2]  # The feature map has shape (1,timeindex, size, size, n_features).
            n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols):  # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0,0,
                                    :, :,
                                    col * images_per_row + row]
                    channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size: (col + 1) * size,  # Displays the grid
                    row * size: (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()


def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    modeled_pre = np.load('predicted-012-0.573.npy')
    #modeled = np.load('predicted.npy')
    accessed = np.load('y.npy')
    plt.plot(modeled_pre, label = "modeled")
    #plt.plot(modeled, label="current modeled")
    plt.plot(accessed+0.05, label = 'accessed')
    plt.legend()
    plt.show()

    model = 'coral_ordinal_lrcn'
    saved_model = 'data/checkpoints/coral_ordinal_lrcn-images.012-0.573.hdf5' #'data/checkpoints/coral_ordinal-features.104-1.384.hdf5' #"data/checkpoints/lstm-features.456-0.148.hdf5" # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 30
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 8
    if model in ['simple', 'lstm', 'coral_ordinal']:
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
