import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy


class TensorBoardReporter(keras.callbacks.Callback):
    def __init__(self, logname='', num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.logname = logname
        self.generated = []
        self.d_loss = []
        self.g_loss = []
        self.x_axis = []
        self.logdir = 'logs/tensorboard/'

    def setLogName(self, name):
        self.logname = name
        self.logdir = "logs/tensorboard/" + name
        self.file_writer = tf.summary.create_file_writer(self.logdir)

    def on_epoch_end(self, epoch, logs=None):
      random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
      generated_images = self.model.generator(random_latent_vectors)
      generated_images.numpy()
      generated_images = numpy.expand_dims(generated_images, axis=3)
      images = numpy.reshape(generated_images, (-1, 128, 128, 1))
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
    def on_train_end(self, logs=None):
      plot = plt.figure()
      f, axarr = plt.subplots(3,3, figsize=(16,16))
      f.suptitle(self.logname + '-OVERVIEW', fontsize=20, fontweight='bold')
      f.tight_layout(rect=[0, 0.05, 1, 0.95])
      i = 0
      for x in range(3):
        for y in range(3):
          axarr[x][y].imshow(numpy.squeeze(self.generated[i][0], axis=(2)))
          axarr[x][y].set_title('Epoch: ' + str(i))
          i = i + 1
      with self.file_writer.as_default():
        tf.summary.image("OVERVIEW", self.plot_to_image(plot), step=0)
      losses = plt.figure()
      plt.plot(self.x_axis, self.d_loss)
      plt.plot(self.x_axis, self.g_loss)
      with self.file_writer.as_default():
        tf.summary.image("LOSSES", self.plot_to_image(losses), step=0)
      
    def plot_to_image(self, figure):    
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      plt.close(figure)
      buf.seek(0)

      digit = tf.image.decode_png(buf.getvalue(), channels=4)
      digit = tf.expand_dims(digit, 0)

      return digit
