
from PIL import Image
from mnist import model
import base64
from flask import Flask, request, render_template,redirect, url_for
import PIL.ImageOps
import PIL.ImageEnhance
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


data = input_data.read_data_sets('data/MNIST/', one_hot=True)
X = data.train.images
Yt = data.train.labels
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def convert_image_to_array():
    img = Image.open('grey.png')
    image_array = np.asarray(img).astype(np.float32)
    # print(image_array)
    return image_array




def covert_image_greyscale():
    img = Image.open('myfile.png').convert('L')
    img = img.resize((28, 28), Image.ANTIALIAS)
    img = PIL.ImageOps.invert(img)
    converter = PIL.ImageEnhance.Color(img)
    img = converter.enhance(50)
    converter2 = PIL.ImageEnhance.Contrast(img)
    img = converter2.enhance(50)
    img.save('grey.png')


@app.route('/')
def hello_world():
    return render_template('upload.html')


@app.route('/perceptron')
def perceptron():
    _xx = convert_image_to_array()
    yy, variables = model.perceptron(_xx.reshape(1, 784))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    saver.restore(sess, "data\\simple_NN_mnist\\MNIST_Perceptron")
    a = sess.run(yy)
    a = np.argmax(a, axis=1)
    sess.close()
    print(a)
    return render_template('hello.html', name='sarthak')


@app.route('/convoluted')
def convol():
    _xx = convert_image_to_array()
    yy, variables,y = model.convoluted_nn(_xx.reshape(1, 784))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    saver.restore(sess, "data\\conv_NN_mnist\\MNIST_conv")
    a = sess.run(yy)
    a = np.argmax(a, axis=1)
    sess.close()
    print(a)
    return render_template('hello.html', name='sarthak')



@app.route('/save', methods=['GET', 'POST'])
def save_image():
    data = request.data
    image_64_decode = base64.decodebytes(data)
    f = open("myfile.png", "wb")
    f.write(image_64_decode)
    f.close()
    covert_image_greyscale()
    return render_template('upload.html')



if __name__ == '__main__':
    app.run(debug='true',host='0.0.0.0' ,port=8080)

