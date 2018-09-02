import keras
from keras.models import Model, load_model
from keras.layers.convolutional import Conv2D, UpSampling2D, SeparableConv2D
from keras.layers import Input, add, Reshape, Activation, BatchNormalization
from bottleneck_residual_blocks import _inverted_residual_block
from keras.applications.mobilenet import relu6

class ContextNet():
    """
    Provides the main structure of contextnet
    """
    def __init__(self, n_labels, image_shape, target_shape):
        self.n_labels = n_labels
        self.image_shape = image_shape
        self.target_shape = target_shape
        pass
    
    def deep_net(self, x):
        #x = Conv2D(32, (3, 3), strides = (2, 2), activation = 'relu', padding = 'same', name = 'conv1', data_format = 'channels_last')(x)
        x = Conv2D(32, (3, 3), strides = (2, 2), padding = 'same', name = 'conv1', data_format = 'channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation(relu6)(x)

        x = _inverted_residual_block(x, 32, (3, 3), t = 1, strides = 1, n = 1)
        x = _inverted_residual_block(x, 32, (3, 3), t = 6, strides = 1, n = 1)
        x = _inverted_residual_block(x, 48, (3, 3), t = 6, strides = 2, n = 3)
        x = _inverted_residual_block(x, 64, (3, 3), t = 6, strides = 2, n =3)
        x = _inverted_residual_block(x, 96, (3, 3), t = 6, strides = 1, n = 2)
        x = _inverted_residual_block(x, 128, (3, 3), t = 6, strides = 1, n = 1)
        #x = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'conv2')(x)
        x = Conv2D(128, (3, 3), padding = 'same', name = 'conv2')(x)
        x = BatchNormalization()(x)
        x = Activation(relu6)(x)

        return x
    
    def shallow_net(self, x):
        """
        x = Conv2D(32, (3, 3), strides = (2, 2), activation = 'relu', padding = 'same', name = 'sep_conv1', data_format = 'channels_last')(x)
        x = SeparableConv2D(64, (3, 3), strides = (2, 2), activation = 'relu', padding = 'same', name = 'sep_conv2')(x)
        x = SeparableConv2D(128, (3, 3), strides = (2, 2), activation = 'relu', padding = 'same', name = 'sep_conv3')(x)
        x = SeparableConv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'sep_conv4')(x)
        """
        x = Conv2D(32, (3, 3), strides = (2, 2), padding = 'same', name = 'sep_conv1', data_format = 'channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation(relu6)(x)

        x = SeparableConv2D(64, (3, 3), strides = (2, 2), padding = 'same', name = 'sep_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation(relu6)(x)

        x = SeparableConv2D(128, (3, 3), strides = (2, 2), padding = 'same', name = 'sep_conv3')(x)
        x = BatchNormalization()(x)
        x = Activation(relu6)(x)

        x = SeparableConv2D(128, (3, 3), padding = 'same', name = 'sep_conv4')(x)
        x = BatchNormalization()(x)
        x = Activation(relu6)(x)

        return x
    
    def feature_fusion_unit(self, input_tensor1, input_tensor2):

        input1 = Conv2D(128, (1, 1), strides=(1, 1), padding = 'same', name = 'unit_conv1')(input_tensor1)
        input2 = UpSampling2D((4, 4))(input_tensor2)
        #input2 = SeparableConv2D(128, (3, 3), strides = (1, 1),  dilation_rate=(4, 4), activation = 'relu', padding = 'same')(input2)
        input2 = SeparableConv2D(128, (3, 3), strides = (1, 1),  dilation_rate=(4, 4), padding = 'same')(input2)
        input2 = BatchNormalization()(input2)
        input2 = Activation(relu6)(input2)

        input2 = Conv2D(128, (1, 1), strides=(1, 1), padding = 'same', name = 'unit_conv2')(input2)
        input_tensors = add([input1, input2])


        result = Conv2D(self.n_labels, (1, 1), strides = (1, 1), activation = 'softmax', padding = 'same', name = 'conv_last')(input_tensors)
        
        return result
    
    def init_model(self):
        
        h, w, d = self.image_shape
        input1 = Input(shape=(h, w, d), name = 'input1')
        output_d = self.deep_net(input1)

        input2 = Input(shape=(int(h/4), int(w/4), d), name = 'input2')
        output_s = self.shallow_net(input2)

        print(f'output1,2: {output_d.shape, output_s.shape}')
        output = self.feature_fusion_unit(output_d, output_s)
        context_net = Model(inputs = [input1, input2], outputs = output, name = 'context_model')
        opt = keras.optimizers.RMSprop(lr = 1e-4, rho=0.9, epsilon=1e-08, decay=0.0)
        context_net.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return context_net



