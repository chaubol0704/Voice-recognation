import os
import cv2
import time
import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

from model import model

class_names =  ['User1', 'User2', 'User3', 'User4', 'None']
pred_file = ['data/user1.jpg','data/user2.jpg','data/user3.jpg','data/user4.jpg']

# đọc file ảnh
def read_image(path):
    # print(index)
    # path = os.path.join(path_root, index[0], index[1])
    # print(path)
    image_p = cv2.imread(path)
    image = cv2.cvtColor(image_p, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return image
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
# name1 = read_file('data/32.wav')

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    audiofile= request.files['audiofile']
    audio_path = "./wav/" + audiofile.filename
    # audio_path = "./wav/file.wav"
    audiofile.save(audio_path)

    image = read_file(audio_path)
    flag = 0
    pred = ''
    img2 = np.array([read_image(image)])
    for i in range(len(pred_file)):
        img1 = np.array([read_image(pred_file[i])])
        distance = classify_images(img1, img2)
        if distance == 0:
            flag = 1
            pred = class_names[i]
    if flag == 0:
        pred = class_names[-1]


    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(image):
        os.remove(image)
    return render_template('index.html', prediction=pred)


if __name__ == '__main__':
    app.run(port=3000, debug=True)