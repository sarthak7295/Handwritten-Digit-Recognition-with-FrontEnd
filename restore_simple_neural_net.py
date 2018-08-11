import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image



data = input_data.read_data_sets('data/MNIST/', one_hot=True)
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))
X = data.train.images
Y = data.train.labels

# defining the imp parameters
n_dim = X.shape[1]
n_classes = 10       #no of classes Mine and Rock so 2
model_path = "D:\\PycharmProjects\\TensorFlow_Models\\MNIST_Perceptron"

# defining the hidden layer and ip and op layer
n_hidden_1 = 120
n_hidden_2 = 120
n_hidden_3 = 120
n_hidden_4 = 120

# Defining my placeholders and variables : input ,weights,biases and output
x = tf.placeholder(tf.float32, [None, n_dim])
y_ = tf.placeholder(tf.float32, [None, n_classes])


def show_image(n, images):
    row_no = n
    a = np.array(images[row_no, :] * 255, dtype=int)
    b = np.reshape(a, (-1, 28))
    img = Image.fromarray(b)
    img.show(title=row_no)


# defining my model
def multilayer_perceptron(x, weights,biases):
    #hidden layer with sigmoid activation

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.sigmoid(layer_1)

    # hidden layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.sigmoid(layer_2)

    # hidden layer 3
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.sigmoid(layer_3)

    # hidden layer 4
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    #output layer
    out_layer = tf.add(tf.matmul(layer_4, weights['out']), biases['out'])
    return out_layer


# defining the weights and biases
# assigns random truncated values to weights and biases
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_classes])),
}

# it is take every neuron has a different bias for it ,  i thought one layer had only on bias, well it
# is all about your personal preference
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_classes])),
}

y = multilayer_perceptron(x,weights,biases)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess,model_path)
show_image(5, X)
prediction_run = sess.run(y, feed_dict={x: X[5].reshape(1, 784)})
print(prediction_run)
