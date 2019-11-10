import tensorflow as tf

import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization, Activation, Add, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, SeparableConv2D, DepthwiseConv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, Nadam, SGD
from keras.callbacks import Callback, ModelCheckpoint

input_shape = (1024, 2048, 3)
batch_size = 32
n_classes = 4
epochs = 30
learning_rate = 0.045
momentum = 0.9

def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
    if(conv_type == 'ds'):
        x = SeparableConv2D(kernel, kernel_size, padding=padding, strides=strides)(inputs)
    else:
        x = Conv2D(kernel, kernel_size, padding=padding, strides=strides)(inputs)  
  
    x = BatchNormalization()(x)
  
    if (relu):
        x = Activation('relu')(x)
  
    return x

def bottleneck(inputs, filters, kernel, t, s, r=False):
    tchannel = K.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

    if r:
        x = Add()([x, inputs])
    
    return x

def bottleneck_block(inputs, filters, kernel, t, strides, n):
    x = bottleneck(inputs, filters, kernel, t, strides)
  
    for i in range(1, n):
        x = bottleneck(x, filters, kernel, t, 1, True)

    return x

def pyramid_pooling_block(input_tensor, bin_sizes):
    concat_list = [input_tensor]
    w = input_tensor.shape[1]
    h = input_tensor.shape[2]
    
    for bin_size in bin_sizes:
        x = AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
        x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = Lambda(lambda x: tf.image.resize(x, (w,h)))(x)

        concat_list.append(x)

    return Concatenate()(concat_list)

input_layer = Input(shape=(320, 480, 3), name='input_layer')

lds_layer = conv_block(input_layer, 'conv', 32, (3, 3), strides=(2, 2))
lds_layer = conv_block(lds_layer, 'ds', 48, (3, 3), strides=(2, 2))
lds_layer = conv_block(lds_layer, 'ds', 64, (3, 3), strides=(2, 2))

gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)

gfe_layer = pyramid_pooling_block(gfe_layer, [2, 4, 6, 8])

ff_layer1 = conv_block(lds_layer, 'conv', 128, kernel_size=(1, 1), padding='same', strides=(1, 1), relu=False)

ff_layer2 = UpSampling2D(size=(4, 4))(gfe_layer)
ff_layer2 = DepthwiseConv2D(128, strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
ff_layer2 = BatchNormalization()(ff_layer2)
ff_layer2 = Activation('relu')(ff_layer2)
ff_layer2 = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(ff_layer2)

ff_final = Add()([ff_layer1, ff_layer2])
ff_final = BatchNormalization()(ff_final)
ff_final = Activation('relu')(ff_final)

classifier = SeparableConv2D(128, kernel_size=(3, 3), padding='same', strides=(1, 1), name='DSConv1_classifier')(ff_final)
classifier = BatchNormalization()(classifier)
classifier = Activation('relu')(classifier)

classifier = SeparableConv2D(128, kernel_size=(3, 3), padding='same', strides=(1, 1), name='DSConv2_classifier')(classifier)
classifier = BatchNormalization()(classifier)
classifier = Activation('relu')(classifier)

classifier = conv_block(classifier, 'conv', 19, (1, 1), strides=(1, 1), padding='same', relu=True)

classifier = Dropout(0.3)(classifier)

classifier = UpSampling2D(size=(8, 8))(classifier)

classifier = Conv2D(n_classes, kernel_size=(1, 1), strides=(1, 1))(classifier)

classifier = Activation('softmax')(classifier)

model = Model(inputs=input_layer, outputs=classifier, name='Fast_SCNN')

optimizer = SGD(momentum=momentum, lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()