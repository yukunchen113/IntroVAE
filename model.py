import tensorflow as tf
import general_utils as gu
from functools import reduce

class VariationalAutoEncoder():
	"""
	This creates a variational auto encoder using tensorflow.
	Create the model and the respective
	"""
	# initialize the parameters
	def __init__(self, inputs, params):
		"""
		Initializes the parameters, and constructs the vae.

		Args:
			inputs: input into the vae encoder.
			params: Parameters for the model

		"""
		# encoder params
		assert reduce(lambda x, y: x or y, [len(l) == 1 or len(l) == 4 for l in params["encoder_layers"]]), "layer sizes must be 3 for conv or size 1 for ff"
		self._encoder_layers = params["encoder_layers"]

		# conv layers in the encoder, extract ones of size 3 and use these first, also if the elements are a list, it is a resnet block
		self._encoder_conv_layers = [l for l in self._encoder_layers if len(l) == 4 or type(l[0]) == list]  

		# ff layers, will extract the ones of len 1 and will use these after the conv layers
		self._encoder_ff_layers = [l for l in self._encoder_layers if len(l) == 1 and not type(l[0]) == list]  

		self._shape_before_flatten = None #this will be updated during runtime. used to reshape the decoder.
		
		# latents params.
		self._latent_size = params["latent_size"] # latent layer size

		# decoder params
		if params["decoder_layers"] is None:
			# if decoder_layers is None, use encoder layers instead.
			self._decoder_layers = self._encoder_layers[::-1]
			self._decoder_layers[-1][0] = inputs.shape[-1]
		else:
			assert reduce(lambda x, y: x or y, [len(l) == 1 or len(l) == 4 for l in params[
				"decoder_layers"]]), "layer sizes must be 3 for conv or size 1 for ff"
			self._decoder_layers = params["decoder_layers"]

		#print(self._decoder_layers)
		self._decoder_ff_layers = [l for l in self._decoder_layers if len(l) == 1 and not type(l[0]) == list]  # ff layers, will extract the ones of len 1 and use these first.
		self._decoder_conv_layers = [l for l in self._decoder_layers if len(l) == 4 or type(l[0]) == list]  # conv layers in the decoder, extract ones of size 3 and will use these after the conv layers
		self._decoder_output_activation = params["decoder_activation"]

		# reconstruction loss params
		self._loss_type = params["loss_type"]
		self.encoder(inputs)

	def get_generation(self, latents_ph=None):
		"""
		For generation/random sampling of the latent space
		Returns latent placeholder to be used and the decoder output.
		:return:
		"""
		if latents_ph is None:
			latents_ph = tf.placeholder(tf.float32, shape=(None, self._latent_size),
										name="latents_ph")  # these are the latent fed into the network	
		
		decoder_out = self.decoder(latents_ph)
		return latents_ph, decoder_out

	def encoder(self, pred, activation=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE):
		"""
		create the encoder

		Args:
			pred: This is the initial input to the encoder, it is the input.

		Returns:
			None
		"""
		latent_size = self._latent_size
		with tf.variable_scope("encoder", reuse=reuse):
			for i in range(len(self._encoder_conv_layers)):
				layer_params = self._encoder_conv_layers[i]
				if type(layer_params[0]) == list:
					initial_pred = pred
					for j in range(len(layer_params)-1):
						pred = tf.layers.conv2d(pred, *layer_params[j], "same", activation=activation)
						if layer_params[j][1] == 1: #layers of size 1 will be used for matching dimensions, and the shortcut will start here.
							initial_pred = pred
					pred = pred + initial_pred
				else:
					pred = tf.layers.conv2d(pred, *layer_params[:-1], "same", activation=activation)
				if not layer_params[-1] is None:
					pred = tf.layers.average_pooling2d(pred, layer_params[-1], layer_params[-1])
				#print("Encoder:", pred.shape, layer_params)
				# apply batch normalization on the features
				#bn_layer = tf.keras.layers.BatchNormalization()
				#pred = bn_layer(pred)


			self._shape_before_flatten = pred.get_shape().as_list()[1:]
			pred = tf.contrib.layers.flatten(pred)

			for i in range(len(self._encoder_ff_layers)):
				cur_activation = activation if not i == len(self._encoder_ff_layers)-1 else lambda x:x
				pred = tf.contrib.layers.fully_connected(pred, *self._encoder_ff_layers[i], activation_fn=cur_activation)

				# apply batch normalization on the features
				#bn_layer = tf.keras.layers.BatchNormalization()
				#pred = bn_layer(pred)

			latent_mean = tf.contrib.layers.fully_connected(pred, latent_size, activation_fn=activation)
			# if we use the log, we can be negative
			latent_log_var = tf.contrib.layers.fully_connected(pred, latent_size, activation_fn=activation)
			noise = tf.random_normal(tf.shape(latent_log_var))

			latent_output = tf.exp(0.5 * latent_log_var) * noise + latent_mean

		return latent_output, [latent_mean, latent_log_var]

	# create the decoder:
	def decoder(self, latent_rep, activation=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE):
		pred = latent_rep
		shape_before_flatten = self._shape_before_flatten
		with tf.variable_scope("decoder", reuse=reuse):
			for i in self._decoder_ff_layers:
				pred = tf.contrib.layers.fully_connected(pred, *i, activation_fn=activation)

				# apply batch normalization on the features
				#bn_layer = tf.keras.layers.BatchNormalization()
				#pred = bn_layer(pred)

			pred = tf.contrib.layers.fully_connected(pred, reduce(lambda x,y: x*y, shape_before_flatten), activation_fn=activation)
			pred = tf.reshape(pred, [-1]+shape_before_flatten)
			for i in range(len(self._decoder_conv_layers)):
				layer_params = self._decoder_conv_layers[i]
				# apply batch normalization on the features
				#bn_layer = tf.keras.layers.BatchNormalization()
				#pred = bn_layer(pred)
				cur_activation = activation if not i == len(self._decoder_conv_layers)-1 else lambda x:x #last layer is regression, so don't use activation
				#print("Decoder", pred.shape, layer_params)
				if not layer_params[-1] is None:
					pred = gu.upscale2d(pred,layer_params[-1]) #upsample if necessary, upsample by padding
				if type(layer_params[0]) == list:
					initial_pred = pred
					for j in range(len(layer_params)-1):
						cur_activation = cur_activation if j == len(layer_params)-1 else activation
						pred = tf.layers.conv2d(pred, *layer_params[j], "same", activation=activation)
						if layer_params[j][1] == 1: #layers of size 1 will be used for matching dimensions, and the shortcut will start here.
							initial_pred = pred
					pred = pred + initial_pred
				else:
					pred = tf.layers.conv2d_transpose(pred, *layer_params[:-1], "same", activation=cur_activation)
			# compress values
			pred = self._decoder_output_activation(pred)

		return pred

	def kl_isonormal_loss(self, latent_mean, latent_log_variance, reduce_sum = True):
		loss = gu.kl_divergence(latent_mean, latent_log_variance)
		if reduce_sum:
			loss = tf.reduce_sum(loss, axis=1)
		return loss

	def reconstruction_loss(self, inputs, decoder_output):
		loss_type = self._loss_type
		#self.TEST=inputs
		return loss_type(inputs, decoder_output)