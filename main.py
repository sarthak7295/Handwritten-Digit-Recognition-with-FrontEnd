
import os
import base64
from flask import Flask, request, render_template,redirect, url_for

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('upload.html')


@app.route('/helloheading')
def index():
    return render_template('hello.html', name='sarthak')


@app.route('/save', methods=['GET', 'POST'])
def save_image():
    data = request.data
    image_64_decode = base64.decodebytes(data)
    f = open("myfile.png", "wb")
    f.write(image_64_decode)
    return render_template('upload.html')



if __name__ == '__main__':
    app.run(debug='true',host='0.0.0.0' ,port=8080)

