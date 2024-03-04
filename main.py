

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
# from keras.applications.inception_v3 import preprocess_input
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

from model import model


def create_spectrogram(filename,name,file_path):
    # print(filename)
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    name = name + '.jpg'
    filename  = os.path.join(file_path, name)
    # print(filename)
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


random.seed(5)
np.random.seed(5)
tf.random.set_seed(5)

# ROOT = "/content/data_mel"

def read_image(path):
    # print(index)
    # path = os.path.join(path_root, index[0], index[1])
    # print(path)
    image_p = cv2.imread(path)
    image = cv2.cvtColor(image_p, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return image




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




def classify_images(face_list1, face_list2, threshold=0.5):
    # Getting the encodings for the passed faces
    tensor1 = model.predict(face_list1)
    tensor2 = model.predict(face_list2)
    # print(tensor1)
    # print(tensor1)
    distance = np.sum(np.square(tensor1-tensor2), axis=-1)
    # print(distance)
    prediction = np.where(distance<=threshold, 0, 1)
    return prediction

class_names = ['User1', 'User2', 'User3', 'User4', 'None']
pred_file = ['data/user1.jpg','data/user2.jpg','data/user3.jpg','data/user4.jpg']

def read_file(file):
  plt.interactive(False)
  clip, sample_rate = librosa.load(file, sr=None)
  fig = plt.figure(figsize=[0.72,0.72])
  ax = fig.add_subplot(111)
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)
  ax.set_frame_on(False)
  S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
  librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
  name = file + '.jpg'
  plt.savefig(name, dpi=400, bbox_inches='tight',pad_inches=0)
  plt.close()
  fig.clf()
  plt.close(fig)
  plt.close('all')
  return name

name1 = read_file('data/32.wav')
name = '32.wav.jpg'
flag = 0
img2 = np.array([read_image(name1)])
for i in range(len(pred_file)):
  img1 = np.array([read_image(pred_file[i])])
  distance = classify_images(img1,img2)
  if distance == 0:
    flag = 1
    print(class_names[i])
if flag == 0:
  print(class_names[-1])
