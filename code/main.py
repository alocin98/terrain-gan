import datatools
import reporter

data = datatools.getData(path='../data/alps_hgt/', resolution=128, scale=1)
reporter = TensorboardReporter()

DCGAN = DCGAN(reporter, 'DCGAN', data, 128)
#GAN = GAN(reporter, 'GAN', data)
#FCCGAN = FCCGAN(reporter, 'FCCGAN', data)
#LSGAN = LSGAN(reporter, 'LSGAN', data)
#StyleGAN = StyleGAN(reporter, 'StyleGAN', data) #https://keras.io/examples/generative/stylegan/

optimizers = [keras.optimizers.Adam(learning_rate=0.0001)]             # Keras optimizers
batch_size = 4
activation_functions = []   
loss_functions = [keras.losses.BinaryCrossentropy()]

DCGAN.train(data, optimizer[0], batch_size, activation_function, loss_function[0], 'DCGAN')
#DCGAN.train()
#FCCGAN.train()
#LSGAN.train()
#StyleGAN.train()