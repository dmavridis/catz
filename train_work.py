#############  Imports   ######################################################

from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, ConvLSTM2D, BatchNormalization
from keras.models import Sequential
from keras.callbacks import Callback
import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os
from PIL import Image
import numpy as np
from keras import backend as K

run = wandb.init(project='catz')

###############   Config variables definition #################################
config = run.config

config.num_epochs = 5
config.batch_size = 32
config.img_dir = "images"
config.height = 96
config.width = 96

val_dir = 'catz/test'
train_dir = 'catz/train'

# automatically get the data if it doesn't exist
if not os.path.exists("catz"):
    print("Downloading catz dataset...")
    subprocess.check_output(
        "curl https://storage.googleapis.com/wandb/catz.tar.gz | tar xz", shell=True)

#################   Util functions ############################################

class ImageCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        validation_X, validation_y = next(
            my_generator(15, val_dir))
        output = self.model.predict(validation_X)
        wandb.log({
            "input": [wandb.Image(np.concatenate(np.split(c, 5, axis=0), axis=2)) for c in validation_X],
            "output": [wandb.Image(np.concatenate([validation_y[i], o], axis=1)) for i, o in enumerate(output)]
        }, commit=False)


def my_generator(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        input_images = np.zeros(
            (batch_size, 5, config.width, config.height, 3))
        output_images = np.zeros((batch_size, config.width, config.height, 3))
        random.shuffle(cat_dirs)
        if ((counter+1)*batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-5]*")
            imgs = [Image.open(img) for img in sorted(input_imgs)]
        
            input_images[i] = np.stack(imgs)
            output_images[i] = np.array(Image.open(
                cat_dirs[counter + i] + "/cat_result.jpg"))
        yield (input_images/255, output_images)
        counter += batch_size



def data_augmentation(input_images, output_images):
    '''
    input images 5 x 96 x 96 x 3
    output image 1 x 96 x 96 x 3
    '''
    
    

def perceptual_distance(y_true, y_pred):
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))




##################  Model definition  #########################################
    
def catz_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters = 32, 
                         kernel_size = 3,
                         activation='relu',
                         use_bias = True,
    #                     recurrent_activation='hard_sigmoid',
                         padding='same',
                         data_format='channels_last',
                         input_shape = ( 5,  config.height, config.width, 3),
                         return_sequences=False,
                         kernel_initializer='random_uniform',
    #                     dropout=0.2,
    #                     recurrent_dropout=0.5
                        ))
#    model.add(BatchNormalization())
    #model.add(MaxPooling2D(2, 2))
    #model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
    
    return model

##################  Model training  ###########################################

model = catz_model()

model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])
model.fit_generator(my_generator(config.batch_size, train_dir),
                    steps_per_epoch=len(
                        glob.glob(train_dir + "/*")) // config.batch_size,
                    epochs=config.num_epochs, callbacks=[
    ImageCallback(), WandbCallback()],
    validation_steps=len(glob.glob(val_dir + "/*")) // config.batch_size,
    validation_data=my_generator(config.batch_size, val_dir))


# https://machinelearningmastery.com/cnn-long-short-term-memory-networks/