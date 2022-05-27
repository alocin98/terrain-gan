data = getData(path='../data/alps_hgt', resolution=128, scale=1)
reporter = TensorboardReporter()

GAN = GAN(reporter, 'GAN', data)
DCGAN = DCGAN(reporter, 'DCGAN', data)
FCCGAN = FCCGAN(reporter, 'FCCGAN', data)
LSGAN = LSGAN(reporter, 'LSGAN', data)
StyleGAN = StyleGAN(reporter, 'StyleGAN', data) #https://keras.io/examples/generative/stylegan/

GAN.train()
DCGAN.train()
FCCGAN.train()
LSGAN.train()
StyleGAN.train()