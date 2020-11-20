from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import TimeDistributed, Activation, LSTM, Dense
from tensorflow.keras import regularizers
from models import ResearchModels
import keras2onnx
import onnx

def coral_ordinal_lrcn_remove_layer(input_shape):
    def add_default_block(model, kernel_filters, init, reg_lambda):
        # conv
        model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                         kernel_initializer=init, kernel_regularizer=regularizers.l2(l=reg_lambda))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        # conv
        model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                         kernel_initializer=init, kernel_regularizer=regularizers.l2(l=reg_lambda))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        # max pool
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        return model

    initialiser = 'glorot_uniform'
    reg_lambda = 0.001

    model = Sequential()

    # first (non-default) block
    model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='valid',
                                     kernel_initializer=initialiser, kernel_regularizer=regularizers.l2(l=reg_lambda)),
                              input_shape=input_shape))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(
        Conv2D(32, (3, 3), kernel_initializer=initialiser, kernel_regularizer=regularizers.l2(l=reg_lambda))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    # 2nd-5th (default) blocks
    model = add_default_block(model, 64, init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 128, init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 256, init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 512, init=initialiser, reg_lambda=reg_lambda)

    # LSTM output head
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=False,
                   input_shape=input_shape,
                   dropout=0.5))
    model.add(Dense(16, activation='relu'))
    #model.add(coral.CoralOrdinal(num_classes=4))  # Ordinal variable has 5 labels, 0 through 4.
    return model

def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'coral_ordinal_lrcn'
    saved_model = 'data/checkpoints/coral_ordinal_lrcn-images.001-3.123.hdf5' #"data/checkpoints/lstm-features.456-0.148.hdf5" # None or weights file
    seq_length = 30
    rm = ResearchModels(4, model, seq_length, saved_model, features_length=14)
    rm.model.summary()
    # Create your new model with the two layers removed and transfer weights
    new_model =  coral_ordinal_lrcn_remove_layer(input_shape=(seq_length, 250, 250, 3))
    new_model.summary()
    def copyModel2Model(model_source, model_target):
        for l_tg, l_sr in zip(model_target.layers, model_source.layers):
            wk0 = l_sr.get_weights()
            l_tg.set_weights(wk0)
        model_target.save('test_complete.hdf5')
        print("model source was copied into model target")

    copyModel2Model(rm.model, new_model)
    onnx_model = keras2onnx.convert_keras(new_model, new_model.name)
    onnx.save(onnx_model, 'test_complete.onnx')

if __name__ == '__main__':
    main()
