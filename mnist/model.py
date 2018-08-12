import tensorflow as tf

def perceptron(X):
    n_dim = 28 * 28
    n_classes = 10
    n_hidden_1 = 120
    n_hidden_2 = 120
    n_hidden_3 = 120
    n_hidden_4 = 120
    def multilayer_perceptron(x, weights, biases):
        # hidden layer with sigmoid activation

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

        # output layer
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

    y = multilayer_perceptron(X, weights, biases)
    return y, [weights, biases]


def convoluted_nn(X):
    x_image = tf.reshape(X, [-1, 28, 28, 1])

    def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
        with tf.variable_scope(name) as scope:
            # Shape of the filter-weights for the convolution
            shape = [filter_size, filter_size, num_input_channels, num_filters]

            # Create new weights (filters) with the given shape
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

            # Create new biases, one for each filter
            biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

            # TensorFlow operation for convolution
            layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

            # Add the biases to the results of the convolution.
            layer += biases

            return layer, [weights,biases]

    def new_pool_layer(input, name):
        with tf.variable_scope(name) as scope:
            # TensorFlow operation for convolution
            layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            return layer

    def new_relu_layer(input, name):
        with tf.variable_scope(name) as scope:
            # TensorFlow operation for convolution
            layer = tf.nn.relu(input)

            return layer

    def new_fc_layer(input, num_inputs, num_outputs, name):
        with tf.variable_scope(name) as scope:
            # Create new weights and biases.
            weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
            biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))

            # Multiply the input and weights, and then add the bias-values.
            layer = tf.matmul(input, weights) + biases

            return layer,[weights,biases]

    # Convolutional Layer 1
    layer_conv1, weights_bias_conv1 = new_conv_layer(input=x_image, num_input_channels=1, filter_size=5, num_filters=6,
                                                name="conv1")

    # Pooling Layer 1
    layer_pool1 = new_pool_layer(layer_conv1, name="pool1")

    # RelU layer 1
    layer_relu1 = new_relu_layer(layer_pool1, name="relu1")

    # Convolutional Layer 2
    layer_conv2, weights_bias_conv2 = new_conv_layer(input=layer_relu1, num_input_channels=6, filter_size=5, num_filters=16,
                                                name="conv2")

    # Pooling Layer 2
    layer_pool2 = new_pool_layer(layer_conv2, name="pool2")

    # RelU layer 2
    layer_relu2 = new_relu_layer(layer_pool2, name="relu2")

    # Flatten Layer
    num_features = layer_relu2.get_shape()[1:4].num_elements()
    layer_flat = tf.reshape(layer_relu2, [-1, num_features])

    # Fully-Connected Layer 1
    layer_fc1 ,weights_bias_fc1= new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")

    # RelU layer 3
    layer_relu3 = new_relu_layer(layer_fc1, name="relu3")

    # Fully-Connected Layer 2
    layer_fc2 ,weights_bias_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=10, name="fc2")
    y_pred = tf.nn.softmax(layer_fc2)
    return y_pred,[layer_conv1, layer_conv2,layer_fc1,layer_fc2],layer_fc2