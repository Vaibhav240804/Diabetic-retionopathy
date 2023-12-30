import json
from flask import Flask, render_template, request, session, redirect
import matplotlib.pyplot as plt
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import load_img,img_to_array
import base64
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import io
import os


app = Flask(__name__)

app.secret_key = 'mysecretkey'

client = MongoClient("mongodb+srv://vkore241:Mumbai#123@cluster0.blfucek.mongodb.net/?retryWrites=true&w=majority")

db = client["first"]
users = db["users"]

UPLOAD_FOLDER = os.path.join('static','uploads')
IMG_SIZE = 256
classes = ['No_DR','Mild','Moderate','Proliferate_DR','Severe']

def get_pred_label(prediction_probabilities):
  print(prediction_probabilities)
  return classes[np.argmax(prediction_probabilities)]

def process_image(image_path):
  """
  Takes an image and turns the image into a Tensor
  """
  # Read in an image file
  image = tf.io.read_file(image_path)

  # Turn the jpg image into a numerical tensor with three color channels (RGB)
  image = tf.image.decode_jpeg(image, channels = 3)

  # Convert the colour channel values from 0-255 values to 0-1 values (Image Representation Using Floating Points Needs 0-1 Values)
  image = tf.image.convert_image_dtype(image, tf.float32) # Normalization !

  # Resize our image to our desired size (224,224)
  image = tf.image.resize(image, size = [IMG_SIZE, IMG_SIZE])

  return image

# Create a simple function to return a tuple (image, label)
def get_image_label(image_path, label):
  """
  Takes an image file path and the associated label
  Processes image and returns tuple of (image, label) 
  """
  image = process_image(image_path)
  return image, label

# Define the batch size, 32 is a good start
BATCH_SIZE = 32

# Create a function to turn our data into batches
def create_data_batches(X, batch_size = BATCH_SIZE):

  print("Creating test data batches...")
  data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) # Only filepaths, no labels (Makes dataset from slices of given tensor)
  data_batch = data.map(process_image).batch(BATCH_SIZE) # Creates batches of given batch size
  return data_batch

def load_model(model_path):
  """
  Loads a saved model from a specified path
  """
  print(f"Loading Saved Model From: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects = {"KerasLayer": hub.KerasLayer})
  return model

model = load_model("./models/best_model.h5")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        image_path = [save_path]
        image = create_data_batches(image_path)
        prediction = model.predict(image)
        pred_label = get_pred_label(prediction)
        print(pred_label)
        # render the template with the results
        # return render_template('result.html', pred_proba=pred_proba, classes=classes, plot_data=plot_data)
        return render_template('predict.html',pred_level =pred_label.replace("_"," ").title())
    else:
        return render_template('predict.html')


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        password = request.form['password']
        bdate = request.form['Bdate']
        mname = request.form['mname']
        register = {
            "fname" :fname,
            "mname":mname,
            "lname" : lname,
            "email" : email,
            "password":password,
            "Bdate":bdate
        }
        try:
            user = users.find_one({'email': email})
            if user:
                return render_template('signup.html', error='user already exists with this email')
            else:
                try:
                    user = users.insert_one(register)
                    session['user_id'] = user['_id']
                    # print(session['user_id'])
                    return redirect('/')
                except:
                    print()
                    return render_template('signup.html', error="some anomally occured")
        except:
            return render_template('signup.html',error='Cannot connect to server,\ntry refreshing page')
       
    else:
        return render_template('signup.html')
    # return render_template('signup.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    render_template("login.html")
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users.find_one({'email': email, 'password': password})
        if user:
            session['user_id'] = user['_id']
            return redirect('/')
        else:
            return render_template('login.html', error='Invalid email or password')

    elif 'user_id' in session and session['user_id']:
        return redirect('/')
    else:
    # handle the case where 'user_id' doesn't exist in session
        return render_template('login.html')
if __name__ == '__main__':
    app.run(debug=True)
