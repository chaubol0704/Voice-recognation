import os
import cv2
import time
import random
import numpy as np
import librosa

import skimage.io
import soundfile as sf
import librosa.display
from IPython.display import Audio
import pandas as pd

import tensorflow as tf
from keras.applications import MobileNetV3Small
from keras.applications.mobilenet_v3 import preprocess_input
# import seaborn as sns
import matplotlib.pyplot as plt
tf.__version__, np.__version__
from keras import backend, layers, metrics
from keras.optimizers import Adam
# from keras.applications import Xception
from keras.models import Model, Sequential
from keras.utils import plot_model
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay



def get_encoder(input_shape):
    """ Returns the image encoding model """

    pretrained_model = MobileNetV3Small(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )

    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.LayerNormalization(),
        layers.Dense(256),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")
    return encode_model


class DistanceLayer(layers.Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


def get_siamese_network(input_shape = (224, 224, 3)):
    encoder = get_encoder(input_shape)

    # Input Layers for the images
    anchor_input   = layers.Input(input_shape, name="Anchor_Input")
    positive_input = layers.Input(input_shape, name="Positive_Input")
    negative_input = layers.Input(input_shape, name="Negative_Input")

    ## Generate the encodings (feature vectors) for the images
    encoded_a = encoder(anchor_input)
    encoded_p = encoder(positive_input)
    encoded_n = encoder(negative_input)

    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    distances = DistanceLayer()(
        encoder(anchor_input),
        encoder(positive_input),
        encoder(negative_input)
    )


    # Creating the Model
    siamese_network = Model(
        inputs  = [anchor_input, positive_input, negative_input],
        outputs = distances,
        name = "Siamese_Network"
    )
    return siamese_network
class SiameseModel(Model):
    # Builds a Siamese model based on a base-model
    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()

        self.margin = margin
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.loss_tracker]


siamese_network = get_siamese_network()
# siamese_network.load_weights('siamese_model.h5')
siamese_network.load_weights('siamese_model1.h5')
siamese_model = SiameseModel(siamese_network)

def extract_encoder(model):
    encoder = get_encoder((224, 224, 3))
    i=0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i+=1
    return encoder
model = extract_encoder(siamese_model)
# def classify_images(face_list1, face_list2, threshold=0.5):
#     # Getting the encodings for the passed faces
#     tensor1 = encoder.predict(face_list1)
#     tensor2 = encoder.predict(face_list2)
#     # print(tensor1)
#     # print(tensor1)
#     distance = np.sum(np.square(tensor1-tensor2), axis=-1)
#     # print(distance)
#     prediction = np.where(distance<=threshold, 0, 1)
#     return prediction

# class_names = ['User1', 'User2', 'User3', 'User4', 'None']
# pred_file = ['/content/test_mel/User1/5.jpg','/content/test_mel/User2/5.jpg','/content/test_mel/User3/5.jpg','/content/test_mel/User4/5.jpg']
#
# def read_file(file):
#   plt.interactive(False)
#   clip, sample_rate = librosa.load(file, sr=None)
#   fig = plt.figure(figsize=[0.72,0.72])
#   ax = fig.add_subplot(111)
#   ax.axes.get_xaxis().set_visible(False)
#   ax.axes.get_yaxis().set_visible(False)
#   ax.set_frame_on(False)
#   S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
#   librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
#   name = file + '.jpg'
#   plt.savefig(name, dpi=400, bbox_inches='tight',pad_inches=0)
#   plt.close()
#   fig.clf()
#   plt.close(fig)
#   plt.close('all')
#   return name
#
#
# flag = 0
# img2 = np.array([read_image(('0', name),path_root = test)])
# for i in range(len(pred_file)):
#   img1 = np.array([read_image(('0', pred_file[i]),path_root = test)])
#   distance = classify_images(img1,img2)
#   if distance == 0:
#     flag = 1
#     print(class_names[i])
# if flag == 0:
#   print(class_names[-1])