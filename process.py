from tkinter import  *
from tkinter import messagebox
import sounddevice as sound
from scipy.io.wavfile import write
import time
import wavio as wv

import os
import cv2

import numpy as np
import librosa
import matplotlib.pyplot as plt


from model import model

class_names =  ['Bol', 'Nat', 'Dan', 'Duoc', 'None']
pred_file = ['data/B5.jpg','data/N5.jpg','data/V5.jpg','data/D5.jpg']

# đọc file ảnh
def read_image(path):
    # print(index)
    # path = os.path.join(path_root, index[0], index[1])
    # print(path)
    image_p = cv2.imread(path)
    image = cv2.cvtColor(image_p, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return image
def probability(distance):
  return 1 / (distance + 1)
def read_file(file):
  plt.interactive(False)
  clip, sample_rate = librosa.load(file, sr=None)
  fig = plt.figure(figsize=[0.72,0.72])
  ax = fig.add_subplot(111)
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)
  ax.set_frame_on(False)
  S = librosa.feature.melspectrogram(y=clip, sr=22050)
  librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
  name = file + '.jpg'
  plt.savefig(name, dpi=400, bbox_inches='tight',pad_inches=0)
  plt.close()
  fig.clf()
  plt.close(fig)
  plt.close('all')
  return name
def classify_distance(face_list1, face_list2, threshold=0.5):
    # Getting the encodings for the passed faces
    tensor1 = model.predict(face_list1)
    tensor2 = model.predict(face_list2)
    # print(tensor1)
    # print(tensor1)
    distance = np.sum(np.square(tensor1-tensor2), axis=-1)
    # print(distance)
    # prediction = np.where(distance<=threshold, 0, 1)
    return distance

def Record():
    # 22050 Hz: Giá trị thường được sử dụng cho âm thanh thoại.
    freq = 22050
    # dur = int(duration.get())
    recording = sound.rec(30*freq, samplerate=freq, channels=2)
#     timer

    try:
        temp= 30
    except:
        print("Please enter ")
    label = Label(text=str(temp), font="arial 40", width=4, background="#4a4a4a")
    label.place(x=240, y=590)
    while temp > 0:
        root.update()
        time.sleep(1)
        temp -=1

        if temp ==0:
            messagebox.showinfo("Time Countdown","Time's up")
        # Label(text=str(temp), font="arial 40",width=4, background="#4a4a4a").pack(x=240, y=590)
        label.config(text=str(temp))
    sound.wait()
    write("recording.wav", freq,recording)

def prediction():
    Record()
    audio_path = "recording.wav"
    # audio_path = "data/30.wav"
    image = read_file(audio_path)
    pred = ''
    distances = []
    img2 = np.array([read_image(image)])
    for i in range(len(pred_file)):
        img1 = np.array([read_image(pred_file[i])])
        distance = classify_distance(img2, img1)
        print(probability(distance))
        distances.append(probability(distance))
        # distances.append(distance)
        # if distance == 0:
        #     flag = 1
        #     pred = class_names[i]
    # if flag == 0:
    #     pred = class_names[-1]
    predict = np.around(max(distances),decimals=4)*100
    pred = class_names[distances.index(max(distances))]
    print(predict)
    print(pred)
    if pred == 'None' or predict < 50:
        txt = " Đây là giọng lạ"
    else:
        txt = "Đây là giọng của: " + pred
    root.update()
    label = Label(text=txt, font="arial 20", bg="#111111",fg="white")
    label.place(x=150, y=480)
    
    if predict >= 35:
        text = 'Độ tương đồng là: ' + str(predict)
        labeltxt = Label(text=text, font="arial 20", bg="#111111", fg="white")
        labeltxt.place(x=150, y=520)
    # if os.path.exists(audio_path):
    #     os.remove(audio_path)
    # if os.path.exists(image):
    #     os.remove(image)
root = Tk()

root.geometry("600x700+400+80")
root.resizable(False,False)
root.title("Voice Recognition")
root.configure(background="#4a4a4a")

#icon
image_icon = PhotoImage(file="Record.png")
root.iconphoto(False,image_icon)

# logo
photo = PhotoImage(file="Record.png")
myimage = Label(image= photo,background="#4a4a4a")
myimage.pack(padx=5,pady=5)

# name
Label(text= "Voice Recognition", font="arial 30 bold", background="#4a4a4a", fg="white").pack()


# entry box
# duration = StringVar()
# entry = Entry(root,textvariable=duration,font="arial 30", width=15).pack(pady=10)

# result =

# button
record = Button(root, font="arial 20", text="Start", bg="#111111",fg="white",border=0,command=prediction).pack(pady=30)

root.mainloop()