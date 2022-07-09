import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy
import math
import io


class TensorBoardReporter(keras.callbacks.Callback):
    def __init__(self, logdir='logs/tensorboard/', logname='', num_img=3, latent_dim=128, print_images=9, checkpoint_filepath = ''):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.logname = logname
        self.generated = []
        self.d_loss = []
        self.g_loss = []
        self.x_axis = []
        self.logdir_root = logdir
        self.logdir = logdir + logname
        self.print_images = print_images

    def setLogName(self, name):
        self.logname = name
        self.logdir = self.logdir_root + name
        self.file_writer = tf.summary.create_file_writer(self.logdir)

    def on_epoch_end(self, epoch, logs=None):
      random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
      generated_images = self.model.generator(random_latent_vectors)
      generated_images.numpy()
      generated_images = numpy.expand_dims(generated_images, axis=3)
      images = numpy.reshape(generated_images, (-1, self.latent_dim, self.latent_dim, 1))
      self.generated.append(images)

      # Save the losses
      self.d_loss.append(logs["d_loss"])
      self.g_loss.append(logs["g_loss"])
      self.x_axis.append(epoch)

      with self.file_writer.as_default():
        tf.summary.image(self.logname, images[0:10], step=epoch)
        tf.summary.scalar('g_loss', logs["g_loss"], step=epoch)
        tf.summary.scalar('d_loss', logs["d_loss"], step=epoch)
      #tf.summary.scalar('loss', log, step=epoch)
      if len(checkpoint_filepath) > 0:
        self.model.generator.save(checkpoint_filepath)
    def on_train_end(self, logs=None):

      rt = int(math.sqrt(self.print_images))

      plot = plt.figure()
      f, axarr = plt.subplots(rt,rt, figsize=(16,16))
      f.suptitle(self.logname + '-OVERVIEW', fontsize=20, fontweight='bold')
      f.tight_layout(rect=[0, 0.05, 1, 0.95])
      i = 0
      for x in range(rt):
        for y in range(rt):
          try:
            axarr[x][y].imshow(numpy.squeeze(self.generated[i][0], axis=(2)))
            axarr[x][y].set_title('Epoch: ' + str(i))
            i = i + 1
          except IndexError:
            break
      with self.file_writer.as_default():
        tf.summary.image("OVERVIEW", self.plot_to_image(plot), step=0)
      losses = plt.figure()
      plt.plot(self.x_axis, self.d_loss)
      plt.plot(self.x_axis, self.g_loss)
      with self.file_writer.as_default():
        tf.summary.image("LOSSES", self.plot_to_image(losses), step=0)
      losses.clf()
      plt.clf()
      plot.clf()
      del plot
      del f
      del axarr
      self.d_loss = []
      self.g_loss = []
      self.x_axis = []
      
    def plot_to_image(self, figure):    
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      plt.close(figure)
      buf.seek(0)

      digit = tf.image.decode_png(buf.getvalue(), channels=4)
      digit = tf.expand_dims(digit, 0)

      return digit
