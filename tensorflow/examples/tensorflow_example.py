# -*- coding: utf-8 -*-
"""
This is a tensorflow example file.
"""
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

def resnet50_example():
    """
    This is a tensorflow example for ResNet50.
    """
    model = ResNet50(weights='imagenet')

    img_path = 'tensorflow/examples/label_image/data/grace_hopper.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])


if __name__ == '__main__':
    resnet50_example()
