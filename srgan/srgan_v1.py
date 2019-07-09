from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras import losses
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

import keras.backend as K

class SRGAN():
    def __init__(self):
        # Input shape
        self.dataset_name = 'celeba'
        self.channels = 3
        self.lr_height = 32                 # Low resolution height
        self.lr_width = 32                  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height*4   # High resolution height
        self.hr_width = self.lr_width*4     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.ids = sorted(os.listdir('celeba-dataset/img_align_celeba/img_align_celeba/'))

        # Number of residual blocks in the generator
        self.n_residual_blocks = 16

        optimizer = Adam(0.0001, 0.9)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        
        print(self.discriminator.summary())

        # Build the generator
        self.generator = self.build_generator()
        
        print(self.generator.summary())

        # High res. and low res. images
        img_lr = Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        self.combined = Model(img_lr, [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', self.mse],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)
        
        print(self.combined.summary())
        
        
    def mse(self, y_true, y_pred):
        y_true = y_true/12.75
        y_pred = y_pred/12.75
        loss = losses.mean_squared_error(y_true, y_pred)
        return loss

    def batches(self, batch_size = 128):
        indices = np.random.randint(0, len(self.ids), batch_size)
        X = np.empty((batch_size, *self.hr_shape), dtype = np.float32)
        Y = np.empty((batch_size, *self.lr_shape), dtype = np.float32)

        for i in range(batch_size):
            hr = image.load_img('celeba-dataset/img_align_celeba/img_align_celeba/'+self.ids[indices[i]],
                                 target_size=self.hr_shape)
            lr = image.load_img('celeba-dataset/img_align_celeba/img_align_celeba/'+self.ids[indices[i]],
                                 target_size=self.lr_shape)
            X[i,] = hr
            Y[i,] = lr

        X = (X - 127.5)/127.5
        Y = Y/255.0

        return X,Y

    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights = None)
        vgg.load_weights('vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5')

        # Download pre-trained VGG19 weights or use the direct one from Keras

        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_shape)

        # Extract image features
        img_features = vgg(img)

        return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = BatchNormalization(momentum=0.8)(d)
            d = PReLU()(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(layer_input)
            u = UpSampling2D(size=2)(u)
            u = PReLU()(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = PReLU()(c1)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d8 = Flatten()(d8)
        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, batch_size=1, sample_interval=500):

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.batches(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size,1))
            fake = np.zeros((batch_size,1))

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch(imgs_lr, [valid, image_features])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

        self.generator.save_weights('gen.h5')
        self.discriminator.save_weights('dis.h5')

    def sample_images(self, epoch):
        r, c = 2, 2

        imgs_hr, imgs_lr = self.batches(batch_size=2)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 127.5 * imgs_lr + 127.5
        imgs_lr = imgs_lr.astype(np.uint8)
        
        fake_hr = 127.5 * fake_hr + 127.5
        fake_hr = fake_hr.astype(np.uint8)
        
        imgs_hr = 127.5 * imgs_hr + 127.5
        imgs_hr = imgs_hr.astype(np.uint8)

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images_%s_%d.png" % (self.dataset_name, epoch))

if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=20000, batch_size=32, sample_interval=500)