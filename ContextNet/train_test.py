import numpy as np 
import cv2
import keras
import data_processing
from keras.models import load_model
import os
from skimage import io
import scipy
from keras.utils.generic_utils import CustomObjectScope
import ContextNet as cn

def train(n_labels, batch_size, epochs, train_steps, val_steps, saved_weight, input_shape, target_shape, train_generator, valid_generator):
    """
    Train the model
    """
    if not os.path.exists(saved_weight):
        #os.makedirs(saved_weight)
        pass

    context_Net = cn.ContextNet(n_labels=n_labels, image_shape=input_shape, target_shape = target_shape)
    model = context_Net.init_model()

    chk = keras.callbacks.ModelCheckpoint(saved_weight + '/weight.h5', monitor='val_loss', verbose=0, save_best_only=True, 
                                                                        save_weights_only=False, mode='auto', period=1)
    
    model.fit_generator(train_generator(batch_size), steps_per_epoch=train_steps, epochs=epochs, verbose=1, callbacks=[chk], 
                                                                        validation_data = valid_generator(batch_size), validation_steps=train_steps)

def build_pridictor(output_last, label_colors, image_shape):
    """
    Get the predicted color map from the output of the model
    """
    labels = np.argmax(output_last, axis=-1)
    labels = labels.reshape(image_shape[0], image_shape[1])
    labels_colored = np.zeros((image_shape[0], image_shape[1], 3)) 

    for label, color in label_colors.items():
        labels_colored[labels == label] = color
  
    #mask = scipy.misc.toimage(labels_colored, mode="RGBA")
    #pred_image = scipy.misc.toimage(image)
    #pred_image.paste(mask, box=None, mask=mask)

    return labels_colored

def test(batch_size, test_steps, target_shape, saved_preImage, saved_weight, label_colors, test_generator):
    """
    Test the model
    """

    if not os.path.exists(saved_preImage):
        os.makedirs(saved_preImage)
        #pass

    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(saved_weight)
        predicted = model.predict_generator(test_generator(batch_size), steps=test_steps)
        for index, image in enumerate(predicted):
            saved_images = os.path.join(saved_preImage, str(index)+'.png')
            pred_image = build_pridictor(image, label_colors, target_shape)
            pred_image = scipy.misc.imresize(pred_image, (256, 512))
            cv2.imwrite(saved_images, pred_image)

def run():
    
    data_dir = './data' #'D:/DeepLearning'  # the path of the dataset
    saved_weight = data_dir + '/weight.h5'  # the path to save the weight of the model
    saved_preImage = os.path.join(data_dir, 'test') # the path to save the predicted results

    image_shape = (256, 512, 3)  # the shape of the input images of the model
    target_shape = (32, 64)  # the shape of outshape of the model
    #train_steps, val_steps = 2500, 450
    batch_size = 1
    epochs = 3 


    train_images, valid_images, test_images, num_classes, label_colors = data_processing.load_data(data_dir)

    train_steps = 5 #int(len(train_images)/batch_size-5)
    val_steps = 5 #int(len(valid_images)/batch_size-5)
    test_steps = 5 #int(len(test_images)/batch_size-5)
    print(train_steps, val_steps, test_steps)

    get_train_image = data_processing.gen_batch_function(train_images, image_shape[0:2])
    get_val_image = data_processing.gen_batch_function(valid_images, image_shape[0:2])
    get_test_image = data_processing.gen_batch_function(test_images, image_shape[0:2], test=True)

    train(num_classes, batch_size, epochs, train_steps, val_steps, data_dir, image_shape, target_shape, get_train_image, get_val_image)
    test(batch_size, test_steps, target_shape, saved_preImage, saved_weight, label_colors, get_test_image)

if __name__ == "__main__":
    run()