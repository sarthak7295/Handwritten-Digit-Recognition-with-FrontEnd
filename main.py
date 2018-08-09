
from PIL import Image
import base64
from flask import Flask, request, render_template,redirect, url_for
import PIL.ImageOps
import PIL.ImageEnhance


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
    f.close()
    covert_image_greyscale()
    return render_template('upload.html')


def covert_image_greyscale():
    img = Image.open('myfile.png').convert('L')
    img = img.resize((28, 28), Image.ANTIALIAS)
    img = PIL.ImageOps.invert(img)
    converter = PIL.ImageEnhance.Color(img)
    img = converter.enhance(50)
    converter2 = PIL.ImageEnhance.Contrast(img)
    img = converter2.enhance(50)
    img.save('grey.png')


if __name__ == '__main__':
    app.run(debug='true',host='0.0.0.0' ,port=8080)

