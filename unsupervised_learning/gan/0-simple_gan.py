#!/usr/bin/env python3
""" simple GAN: generator and discriminator playing an adversarial game """
import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """ GAN trained with least squares losses (real -> 1, fake -> -1) """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """ stores the networks and sets their losses and optimizers """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # generator wants the discriminator to output 1 on fakes
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # discriminator wants 1 on reals (x) and -1 on fakes (y)
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape)))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """ generates a fake sample of size batch_size """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """ picks a random real sample of size batch_size """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """ disc_iter discriminator updates, then one generator update """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)
                discr_loss = self.discriminator.loss(
                    self.discriminator(real_sample, training=True),
                    self.discriminator(fake_sample, training=True))
            grads = tape.gradient(discr_loss,
                                  self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            gen_loss = self.generator.loss(
                self.discriminator(fake_sample, training=False))
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
