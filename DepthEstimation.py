
import sys
import os
from urllib.request import urlopen
import io
import shutil
import stat

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.slim as slim;

import nengo
import nengo_dl
# Load the Drive helper and mount
from google.colab import drive
# This will prompt for authorization.
drive.mount('/content/drive')


from pylab import *
import os
import sys
#from keras_contrib.applications import densenet
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.python.keras.engine import Layer
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.models import *
#from keras.applications.imagenet_utils import _obtain_input_shape
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#from keras.models import Model

def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = permute_dimensions(X, [0, 3, 1, 2])
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
		
		
def FCN_Vgg16_32s(input_shape=(640, 640, 3), weight_decay=0., batch_momentum=0.9, batch_shape=(1,) + (640,640) + (3, ), classes=12):
    #x = x1.flatten()
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
        print("x", img_input)
        print("y", image_size)
    else:
        img_input = Input(shape=input_shape)
        #img_input = "/content/drive/My Drive/Colab Notebooks/img1.jpeg"
        image_size = input_shape[0:2]
        print("x", img_input)
        print("y", image_size)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)
    #print(x)
    #x = .add(Dense(10, activation='softmax'))
    #x = tf.keras.layers.Dense((tf.math.argmax(tf.Variable(x), dimension=3, name="Pred")))(x)
    #x = np.argmax(np.squeeze(x), axis=-1).astype(np.uint8)
    #x = keras.layers.Dense(1)(x)
    #x = tf.keras.backend.argmax(x, axis=-1)
    #x = (lambda y : K.argmax(y, axis=-1))(x)
    x = Flatten()(x)
    model = Model(img_input,x)
    #weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    #model.load_weights(weights_path, by_name=True)
    #print(model.summary())
    return model

def preprocess_img(img):
    x = image.load_img(img, target_size=(640,640))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
	
	
#Create Keras Model - Already defined in Keras. Train in nengo_dl
image_shape = (640, 640)

model = FCN_Vgg16_32s()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_weights = "/content/drive/My Drive/Colab Notebooks/apc_weights.hdf5"



class KerasNode:
    def __init__(self, keras_model):
        self.model = keras_model

    def pre_build(self, *args):
        self.model = keras.models.clone_model(self.model)

    def __call__(self, t, x):
        images = tf.reshape(x, (-1,) + image_shape)
        print(images)
        return self.model.call(images)

    def post_build(self, sess, rng):
        self.model.load_weights(model_weights)
		
		
#CREATE TENSOR NODE
train_list = ["/content/drive/My Drive/Colab Notebooks/img1.jpg"]
class_names = ['background','crayola_24_ct','expo_dry_erase_board_eraser','folgers_classic_roast_coffee','scotch_duct_tape','up_glucose_bottle','laugh_out_loud_joke_book','soft_white_lightbulb','kleenex_tissue_box','dove_beauty_bar','elmers_washable_no_run_school_glue','rawlings_baseball']
num_classes = 4915200
image_shape = (640,640,3)
net_input_shape = np.prod(image_shape)
for x in train_list:
    x = preprocess_img(x)
    with nengo.Network() as net:
        input_node = nengo.Node(output = x.flatten())
        keras_node = nengo_dl.TensorNode(KerasNode(model), size_in=net_input_shape, size_out = num_classes)
        # connect up our input to our keras node
        nengo.Connection(input_node, keras_node, synapse=None)
        #nengo.Connection()
        keras_p = nengo.Probe(keras_node)
		
test_list = ["/content/drive/My Drive/Colab Notebooks/img1.jpg"]
minibatch_size = 1
np.random.seed(1)
for i in test_list:
    test_inputs = preprocess_img(i)
    test_inputs = test_inputs.reshape((-1, net_input_shape))
    test_inputs = test_inputs[:, None, :]

with nengo_dl.Simulator(net, minibatch_size=1) as sim:
    sim.step(data={input_node:test_inputs})
	
tensornode_output = sim.data[keras_p]
output = np.reshape(tensornode_output, [640,640,12])
output = np.argmax(output, axis=2)
output_c = np.subtract(255,np.multiply(20,output))
imgplot = plt.imshow(output_c)

img = image.load_img(test_list[0], target_size=(640,640))
x1 = image.img_to_array(img)
plt.imshow(x1.astype(int))