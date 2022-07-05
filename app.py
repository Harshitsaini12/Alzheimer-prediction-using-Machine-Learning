from flask import Flask, render_template ,request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

app = Flask(__name__)
CLASS_NAMES=['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
NUM_CLASSES=4
BASE_TRAIN_DIR="Alzheimer_s Dataset\\train"
BASE_TEST_DIR="Alzheimer_s Dataset\\test"
AUTOTUNE=tf.data.experimental.AUTOTUNE
NR_EPOCHS=50



new_moddel=load_model("alzheimer_model.h5")


@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/',methods=['POST'])
def predict():
    imageFILE=request.files['imagefile']
    image_path="./images/"+ imageFILE.filename
    imageFILE.save(image_path)

    image=tf.keras.preprocessing.image.load_img(image_path)
    image=tf.image.resize(image,(224,224))
    input_array=tf.keras.preprocessing.image.img_to_array(image)
    input_array=np.array([input_array])
    pred=new_moddel.predict(input_array)
    res=np.argmax(pred)
    re1=CLASS_NAMES[res]
    return render_template('index.html' , prediction=re1)


if __name__=="__main__":
    app.run(debug=True)