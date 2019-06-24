#this file contains the various architectures I used throughout the experiments.

"""
#this was used when testing the loss.
#create the encoder:
def encoder(pred, activation=tf.nn.relu, latent_size=32):
	with tf.variable_scope("encoder"):
		#we will run this architecture later on, mnist is too small to run these
		
		pred = tf.layers.conv2d(pred, 64, 4, 2, "same", activation=activation)
		pred = tf.layers.conv2d(pred, 64, 4, 2, "same", activation=activation)
		shape_before_flatten = tf.shape(pred)

		pred = tf.contrib.layers.flatten(pred)
		pred = tf.layers.dense(pred,256)
		
		latent_mean = tf.layers.dense(pred, latent_size)

		#if we use the log, we can be negative
		latent_log_std = tf.layers.dense(pred, latent_size)
		noise = tf.random_normal(tf.shape(latent_log_std))
		latent = tf.exp(0.5*latent_log_std)*noise+latent_mean
	return latent, shape_before_flatten, [latent_mean, latent_log_std]


#create the decoder:
def decoder(latent_rep, shape_before_flatten, activation=tf.nn.relu):
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
		pred = tf.layers.dense(latent_rep, 256)
		pred = tf.layers.dense(pred, 7*7*64)#this is hard coded right now
		pred = tf.reshape(pred, shape_before_flatten)

		pred = tf.layers.conv2d_transpose(pred, 64, 4, 2, "same", activation=activation)
		
		#we will run this architecture later on, mnist is too small to run these
		pred = tf.layers.conv2d_transpose(pred, 1, 4, 2, "same", activation=activation)

		#compress values to be between 0 and 1
		#pred = tf.sigmoid(pred)
		
	return pred
"""