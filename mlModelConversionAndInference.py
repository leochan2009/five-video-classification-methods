import os
import tensorflow as tf
import numpy as np
import coral_ordinal as coral
import pandas as pd
from data import DataSet
import random
from models import ResearchModels

coral_ordinal_weigth = np.array([[ -0.8586888909339905],[ 0.9852843284606934],[0.5865223407745361],[0.5686368346214294],
                        [0.7041797041893005], [-0.2871415317058563],[-0.2728332579135895],[-0.7504894733428955],
                        [0.5025442838668823], [0.4780637323856354], [-0.47675463557243347],[0.4587705135345459],
                        [-1.2251392602920532],[2.215362548828125],[1.268646001815796],[-0.08366962522268295]])

coral_ordinal_bias = np.array([7.2205023765563965, -1.2660282850265503,-9.123575210571289])

def convertToTFLiteModel(modeldir, name):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_saved_model(modeldir)
    tflite_model = converter.convert()
    # Save the model.
    with open(os.path.join(modeldir, name), 'wb') as f:
      f.write(tflite_model)

def saveModelWithInputShapeModification(modelDir):
    model = 'coral_ordinal_lrcn'
    saved_model = 'data/checkpoints/coral_ordinal_lrcn-images.355-0.294-val_loss-0.179.hdf5'  # "data/checkpoints/lstm-features.456-0.148.hdf5" # None or weights file
    seq_length = 30
    from models import ResearchModels
    rm = ResearchModels(4, model, seq_length, saved_model, features_length=14)
    import tensorflow as tf
    from mlModelTrim import coral_ordinal_lrcn_remove_layer, copyModel2Model
    new_model = coral_ordinal_lrcn_remove_layer(input_shape=(seq_length, 250, 250, 3))
    copyModel2Model(rm.model, new_model)
    new_model.summary()
    run_model = tf.function(lambda x: new_model(x))
    BATCH_SIZE = 1
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, seq_length, 250, 250, 3], new_model.inputs[0].dtype))
    new_model.save(modelDir, save_format="tf", signatures=concrete_func)


def getSequenceFrame(dataObj, trainData):
    # Get a random sample.
    sample = random.choice(trainData)
    # Check to see if we've already saved this sequence.
    # Get and resample frames.
    frames = dataObj.get_frames_for_sample(sample)
    frames = dataObj.rescale_list(frames, dataObj.seq_length)

    # Build the image sequence
    sequence = dataObj.build_image_sequence(frames)

    return np.array([sequence]), dataObj.get_class_ordinal_encode(sample[1])


dataObj = DataSet(
    seq_length=30,
    class_limit=None,
    image_shape=(250, 250, 3), labelEncoding='coral_ordinal'
)
trainData, test = dataObj.split_train_test()

def runWithConvertedModel(interpreter, input_data):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    fc_inputs = tf.matmul(output_data, coral_ordinal_weigth)
    logits = fc_inputs + coral_ordinal_bias
    tensor_probs = coral.ordinal_softmax(logits)
    # Convert the tensor into a pandas dataframe.
    probs_df = pd.DataFrame(tensor_probs.numpy())
    probs_df.head()
    labels = probs_df.idxmax(axis=1)

    return labels.values[0]

def runWithOriginalModel(model, inputdata):
    ordinal_logits = model.predict(inputdata)
    # Convert from logits to label probabilities. This is initially a tensorflow tensor.
    tensor_probs = coral.ordinal_softmax(ordinal_logits)

    # Convert the tensor into a pandas dataframe.
    probs_df = pd.DataFrame(tensor_probs.numpy())

    probs_df.head()
    labels = probs_df.idxmax(axis=1)
    return labels.values[0]

def main():
    modeldir = "saved_model"
    convertedModelName = 'model.tflite'
    #saveModelWithInputShapeModification(modeldir)
    #convertToTFLiteModel(modeldir, convertedModelName)
    modelPath = 'data/checkpoints/coral_ordinal_lrcn-images.355-0.294-val_loss-0.179.hdf5'
    rm = ResearchModels(len(dataObj.classes), "coral_ordinal_lrcn", 30, modelPath, features_length=14)
    interpreter = tf.lite.Interpreter(model_path=os.path.join(modeldir, convertedModelName))
    interpreter.allocate_tensors()
    for i in range(200):
        input_data, groundtruth = getSequenceFrame(dataObj, trainData)
        pred = runWithOriginalModel(rm.model, input_data)
        pred_converted = runWithConvertedModel(interpreter, input_data)
        print('original Model, converted Model, groundtruth: ', pred, pred_converted, groundtruth)

if __name__ == '__main__':
    main()