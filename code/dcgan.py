from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from gan import GAN

def DCGAN(reporter, title, latent_dim):
    gan = GAN(reporter, title, latent_dim)
    generator = keras.Sequential(
        [
            keras.Input(shape=(self.latent_dim,)),
            layers.Dense(8 * 8 * 128),
            layers.Reshape((8, 8, 128)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(1024, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name="generator",
        )
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(128, 128,1)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
        )
    gan.setGenerator(generator)
    gan.setDiscriminator(discriminator)
    return gan