import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import random


def show_n_random_images(n, images):
    for counter in range(n):
        row_no = random.randint(1,500)
        a = np.array(images[row_no,:]*255, dtype=int)
        b = np.reshape(a, (-1, 28))
        img = Image.fromarray(b)
        img.show()


def show_image(n, images):
    row_no = n
    print(images[row_no, :])
    a = np.array(images[row_no,:] * 255, dtype=int)
    b = np.reshape(a, (-1, 28))
    img = Image.fromarray(b)
    img.show(title=row_no)

# show_n_random_images(3, X)
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))
# print(data.train.images)
X = data.train.images
show_image(5000,X)


