import os
from flask import Flask, render_template, request, jsonify, url_for, send_file, send_from_directory
from keras.utils import load_img
from keras.utils import img_to_array
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import shutil
import time

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './img/'

# model = tf.keras.models.load_model('./VGG19_24_6_1.h5')
model = tf.keras.models.load_model('./VGG19_24_6_30.h5')
target_names = ['Normal','Pneumonia']

def is_grey_scale(img_path):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        for i in range(w):
            for j in range(h):
                r, g, b = img.getpixel((i,j))
                if r != g != b: 
                    return False
        return True

@app.route('/', methods = ['GET'])
def home_page():
    return render_template("home.html")

@app.route('/classification', methods = ['GET'])
def classification_page():
    return render_template('klasifikasi.html')

@app.route('/hasil', methods = ['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)

    imagefile.save(image_path)

    print(load_img(image_path).size)
    height, width = load_img(image_path).size

    convert_success = False
    image = load_img(image_path, target_size=(224,224))
    convert_success = True

    converted_filename = 'converted_'+imagefile.filename
    image.save('./img/'+converted_filename)

    is_grayscale = is_grey_scale(image_path)

    # image_size = len(imagefile.read())
    image_size = os.path.getsize(image_path)
    image_size_kb = image_size/1024
    image_size_mb = image_size_kb/1024

    print(f"Size Image : {image_size_mb}")

    if image_size_mb <= 2:
        image_size_ver = True
    else:
        image_size_ver = False

    if is_grayscale == True:
        image1 = img_to_array(image)
        image1 = np.expand_dims(image1, axis=0)
        image1 = np.vstack([image1])
        classification_success = False
        start_time = time.time()
        prediksi = model.predict(image1)
        processing_time = time.time() - start_time
        # logit_layer = model.layer[-1]
        # logits = logit_layer(image1)
        # print(f"Logits : {logits}")
        classification_success = True
        skor = np.max(prediksi)
        persentase = "{:.2f}%".format(skor*100)
        print("{:.2f}%".format(np.min(prediksi)*100))
        print(prediksi)
        print(persentase)
        print(skor)
        print('----------')
        classes = np.argmax(prediksi)
        hasil = target_names[classes]
        print(prediksi[0][0])
        # normal_score : prediksi[0][0]
        # print(f"normal_score : {normal_score}")
        # pneumonia_score : prediksi[0][1]
        # print(f"pneumonia_score : {pneumonia_score}")
        print(hasil)
    else:
        persentase = "-"
        classification_success = False
        prediksi = [[0,0]]
        processing_time = 0
        hasil = 'Gambar tidak terdeteksi sebagai citra Sinar X'

    return render_template(
        "hasil.html",
        convert_success = convert_success,
        # image_size_ver_html = image_size_ver,
        height_html = height,
        width_html = width,
        result = hasil,
        is_grayscale = is_grayscale,
        skor = persentase,
        classification_success = classification_success,
        classification_time = round(processing_time, 2),
        normal_score_html = prediksi[0][0],
        pneumonia_score_html = prediksi[0][1],
        img = imagefile.filename,
        converted_img = converted_filename
        )

@app.route('/img/<fileimg>')
def send_uploaded_image(fileimg=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], fileimg)

@app.route('/img/<fileimg2>')
def send_converted_image(fileimg2=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], fileimg2)
    
if __name__ == '__main__':
    app.run(port=3000, debug=True)