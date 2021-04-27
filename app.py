from flask import Flask, request, redirect, url_for,render_template, flash
import cv2
import numpy as np 
from werkzeug.utils import secure_filename
from PIL import Image
import os
import sys
from app_helper import *



app = Flask(__name__,static_url_path="/static")

UPLOAD_FOLDER ='static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'
ALLOWED_EXTENSIONS = {'jpg', 'png','.jpeg','.mp4'}


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024 

def allowed_file(filename):
     return '.' in filename and filename.rsplit('.', 1)[1].lower()      in ALLOWED_EXTENSIONS

@app.route("/")
def index():
  return render_template("index.html")
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      # create a secure filename
      filename = secure_filename(f.filename)
      print(filename)
      # save file to /static/uploads
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      print(filepath)
      f.save(filepath)
      get_image(filepath, filename)
      
      return render_template("uploaded.html", display_detection = filename, fname = filename)      


if __name__ == "__main__":
    app.run(debug=True, port=8000)