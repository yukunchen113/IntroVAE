import tensorflow as tf
import numpy as np
import general_constants as gc
import general_utils as gu
import os
import time
import matplotlib.pyplot as plt
from functools import reduce
import scipy.ndimage as ndi
import shutil
import params as pm
from PIL import Image
from PIL import ImageDraw
import model as md
from PIL import ImageFont
np.set_printoptions(suppress=True)

def main(is_train=False, **kwargs):
	tf.reset_default_graph()
	params = pm.general_params
	batch_size = params["batch_size"]
	log_step = params["log_step"]
	num_steps = params["num_steps"]
	learning_rate = params["learning_rate"]
	plot_step = params["plot_step"]
	validation_step = params["validation_step"]
	latent_size = params["latent_size"]
	initializer_step = params["initializer_step"]
	imreshape_size = params["imreshape_size"]
	validation_tolerence = params["validation_tolerence"]
	data_shape = params["data_shape"]

	load_prev = kwargs["load_prev"]
	start_step = 0
	if not is_train:
		#when testing, the following keyword arguments must be specified.
		assert "test_images" in kwargs or "test_latents" in kwargs, "must specify a latent space or test images for reconstruction"
		assert "modelsavedir" in kwargs, "must specify path for saved models"
		if "test_images" in kwargs:
			test_images = kwargs["test_images"]
			if test_images.ndim < 4:
				test_images = np.expand_dims(test_images, 0)
		else:
			test_images = None

		if "test_latents" in kwargs:
			test_latents = np.asarray(kwargs["test_latents"])
			if test_latents.ndim < 2:
				test_latents = np.expand_dims(test_latents, 0)
		else:
			test_latents = None
		for i in kwargs["modelsavedir"]:
			assert "checkpoint" in os.listdir(i), "no saved model in one of the specified directories:%s"%i
		modelsavedir = [os.path.join(i, "model.ckpt") for i in kwargs["modelsavedir"]]

	if "max_steps" in kwargs:
		num_steps = kwargs["max_steps"]

	if is_train:
		# if specified, create general path (if doesn't exist) 
		# Then, define model save path
		if "save_folder" in kwargs:
			pm.create_new_path(pm.logdir, False)
			logdir = os.path.join(pm.logdir, kwargs["save_folder"])
		else:
			logdir = pm.logdir

		# delete previous model path if training.
		pm.create_new_path(logdir, is_train and not load_prev)

		imgdir = os.path.join(logdir, "images")
		# delete previous image path if training.
		pm.create_new_path(imgdir, is_train and not load_prev)
		if load_prev:
			start_step = [int(i[:-4].split("_")[-1]) for i in os.listdir(imgdir)]
			start_step.sort()
			start_step = start_step[-1]

		tblogdir = os.path.join(logdir, "log")
		# delete previous tensorboard log path if training.
		pm.create_new_path(tblogdir, is_train and not load_prev)

		
		modelsavedir = os.path.join(logdir, "model_save_point")
		pm.create_new_path(modelsavedir, False)

		log_file = os.path.join(logdir, "log.txt")
		if os.path.exists(log_file) and (is_train and not load_prev):
			os.remove(log_file)

		# copy parameters to predictions.
		if not load_prev:
			for file_to_be_copied in ["params.py", "model.py", "main.py"]:
				shutil.copyfile(file_to_be_copied, os.path.join(logdir, file_to_be_copied))

	# get data
	if is_train:
		dataset, get_group = gu.get_celeba_data(gc.datapath)#, preprocess_fn = lambda x: np.array(Image.fromarray(x).resize((x.shape[0], *imreshape_size, x.shape[-1]))))
		group_size = dataset.get_group_size()
		test_images, test_labels = get_group(group_num=initializer_step//group_size, random_selection=False, remove_past=True)  # get images, and remove from get_group iterator
		validation_images, validation_labels = get_group(group_num=1, random_selection=False, remove_past=True)  # get images, and remove from get_group iterator

	data_shape = data_shape if test_images is None else list(test_images.shape)
	data_shape[0] = batch_size

	# create placeholders
	inputs_ph = tf.placeholder(tf.float32, shape=(None, *data_shape[1:]), name="inputs_ph")  # these are the ones fed into the network, after been batched.
	outputs_ph = tf.placeholder(tf.float32, name="outputs_ph")  # these are the ones fed into the network, after been batched.
	#inputs_set_ph = tf.placeholder(tf.float32, name="inputs_set_ph")  # these are the ones fed into the iterator, to be batched.
	#outputs_set_ph = tf.placeholder(tf.float32, name="outputs_set_ph")  # these are the ones fed into the iterator, to be batched.
	#iterator, next_element = gu.get_iterator(batch_size, inputs=inputs_set_ph, labels=outputs_set_ph)  # this is the iterator.

	#####################
	### preprocessing ###
	#####################
	
	inputs = inputs_ph
	# crop to 128x128 (centered), this number was experimentally found
	"""
	image_crop_size = [128,128]
	inputs=tf.image.crop_to_bounding_box(inputs, 
		(inputs.shape[-3]-image_crop_size[0])//2,
		(inputs.shape[-2]-image_crop_size[1])//2,
		image_crop_size[0],
		image_crop_size[1],
		)
	"""
	inputs = (tf.image.resize_images(inputs, imreshape_size, True))
	inputs = params["preprocess_inputs"](tf.clip_by_value(inputs, 0, 255))
	
	##################
	### make model ###
	##################
	# Create VAE
	vae = md.VariationalAutoEncoder(inputs, pm.model_params)
	latent_output, dist_params = vae.encoder(inputs)
	reconstruction_prediction = vae.decoder(latent_output)
	latents_ph, generation_prediction = vae.get_generation() # just runs decoder and returns placeholder of appropriate size.
	#stop gradient for reconstruction.
	#stopped_grad_reconstruction_prediction = vae.decoder(tf.stop_gradient(latent_output))
	
	# create "GAN" portion of the IntroVAE
	generator_energy_reconstruction, generator_er_dist_params = vae.encoder(reconstruction_prediction)
	generator_energy_generation, generator_eg_dist_params = vae.encoder(generation_prediction)
	# same as above with stopped gradients for inference
	inference_energy_reconstruction, inference_er_dist_params = vae.encoder(tf.stop_gradient(reconstruction_prediction))
	inference_energy_generation, inference_eg_dist_params = vae.encoder(tf.stop_gradient(generation_prediction))
	
	#for i in tf.global_variables():
	#	print(i.name)

	##################
	### get losses ###
	##################
	get_energy = lambda dist_p: tf.reduce_mean(vae.kl_isonormal_loss(*dist_p))

	# VAE losses
	reconstruction_loss = tf.reduce_mean(tf.abs(vae.reconstruction_loss(inputs, reconstruction_prediction)))/2
	regularization_loss = get_energy(dist_params)
	
	#kl_multiplier = tf.placeholder(tf.float32, name="kl_multiplier")
	#loss = reconstruction_loss+kl_multiplier*regularization_loss

	# IntroVAE losses
	# hyperparameters
	alpha_multiplier = tf.placeholder(tf.float32, name="alpha_multiplier")
	disc_offset = tf.placeholder(tf.float32, name="disc_offset")
	reconstruction_multiplier = tf.placeholder(tf.float32, name="reconstruction_multiplier")

	# inference loss calculations
	inference_reconstruction_energy = get_energy(inference_er_dist_params)
	inference_generation_energy = get_energy(inference_eg_dist_params)
	introvae_inference_loss = regularization_loss + alpha_multiplier*( #updates the encoder
		tf.maximum((disc_offset - inference_reconstruction_energy), 0) +
		tf.maximum((disc_offset - inference_generation_energy), 0)) + reconstruction_loss*reconstruction_multiplier

	# generation loss calculations
	generator_reconstruction_energy = get_energy(generator_er_dist_params)
	generator_generation_energy = get_energy(generator_eg_dist_params)
	introvae_generator_loss = alpha_multiplier*( # updates the decoder
			generator_reconstruction_energy +
			generator_generation_energy) + reconstruction_loss*reconstruction_multiplier

	loss = introvae_generator_loss + introvae_inference_loss

	################
	### training ###
	################
	opt = tf.train.AdamOptimizer(learning_rate)
	###########
	#### WARNING! Should check if the stop gradients work
	###########
	inference_op = opt.minimize(introvae_inference_loss, var_list=[v for v in tf.global_variables() if v.name.startswith("encoder")])
	generator_op = opt.minimize(introvae_generator_loss, var_list=[v for v in tf.global_variables() if v.name.startswith("decoder")])

	#save model:
	saver = tf.train.Saver()

	# latent space analysis:
	
	# min, max means and standard deviations across the 
	#std_analysis = [tf.reduce_mean(tf.exp(0.5*dist_params[1])), tf.reduce_min(tf.exp(0.5*dist_params[1])), tf.reduce_max(tf.exp(0.5*dist_params[1]))]
	#mean_analysis = [tf.reduce_mean(dist_params[0]), tf.reduce_min(dist_params[0]), tf.reduce_max(dist_params[0])]

	#latent_element_std_average = tf.reduce_mean(tf.exp(0.5*dist_params[1]), axis=0)
	#latent_element_mean_average = tf.reduce_mean(dist_params[0], axis=0)
	#latent_element_kld_average = tf.reduce_mean(vae.kl_isonormal_loss(*dist_params, False), axis=0)
	#latent_element_max = tf.reduce_max(latent_output, axis=0)
	#latent_element_min = tf.reduce_min(latent_output, axis=0)

	#latent_analysis_package = [latent_element_mean_average, 
	#						latent_element_std_average, 
	#						latent_element_kld_average,
	#						latent_element_max,
	#						latent_element_min]

	# run model
	with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))) as sess:
		# print(training_data["data"].shape)
		sess.run(tf.global_variables_initializer())
		if is_train:
			#test_images = test_images[:batch_size]

			validation_feed_dict = {  # keep all these images to test constant.
				inputs_ph: validation_images,
				outputs_ph: validation_labels,
			}
			prev_validation_loss = None
			current_validation_count = 0
		else:
			num_steps = len(modelsavedir)
			return_dict = {}
		first_step = True
		if is_train:
			log_file = open(log_file, "a")

		#contents to hold across training
		disc_offset_val_past = np.zeros((300)) #keep the values for the past 300 steps
		regularization_loss_val = 0

		for step in range(start_step,num_steps):
			if not (is_train and not load_prev):
				if not is_train:
					saved_dir = modelsavedir[step]
				else:
					saved_dir = os.path.join(modelsavedir, "model.ckpt")
				saver.restore(sess, saved_dir)
			if is_train:
				#print(not step%(initializer_step//batch_size))
				if not step%(initializer_step//batch_size):
					#get images and labels
					images, labels = get_group(group_num=initializer_step//group_size, random_selection=True)
					images = images[:images.shape[0]//batch_size*batch_size] #cut to batch_size
					labels = labels[:labels.shape[0]//batch_size*batch_size]
					images, labels = gu.shuffle_arrays(images, labels)


				current_batch = step%(initializer_step//batch_size)
				#print(current_batch, len(images))
				feed_dict = {
					inputs_ph:images[batch_size*(current_batch):batch_size*(current_batch+1)],
					outputs_ph:labels[batch_size*(current_batch):batch_size*(current_batch+1)],
					latents_ph:np.random.normal(size=[batch_size, latent_size])
				}

				#test_val_a= 0
				#print("!%d"%test_val_a)
				#for i in feed_dict.values():
				#	print(i.shape)

				train_feed = feed_dict.copy()

				if not "regularization_function" in kwargs:
					alpha_multiplier_val = 0.3
					disc_offset_val = 230.0 # should be selected to be a little large than max(KLD)
					reconstruction_multiplier_val = 1.0
				else:
					alpha_multiplier_val, disc_offset_val, reconstruction_multiplier_val = kwargs["regularization_function"](step=step, 
						disc_offset_val_past=disc_offset_val_past)


				disc_offset_val_past[1:] = disc_offset_val_past[:-1]
				disc_offset_val_past[0] = max(disc_offset_val, regularization_loss_val)


				#print("\n\nHERE", type(disc_offset_val))
				#print("HERE", type(reconstruction_multiplier_val))
				train_feed[alpha_multiplier] = alpha_multiplier_val
				train_feed[disc_offset] = disc_offset_val
				train_feed[reconstruction_multiplier] = reconstruction_multiplier_val
				#test_val_a+=1
				#print("!%d"%test_val_a)

				#test to see variable changes pre and post training:
				#encoder_example_test_variable = [v for v in tf.global_variables() if v.name.startswith("encoder")][0]
				#decoder_example_test_variable = [v for v in tf.global_variables() if v.name.startswith("decoder")][0]
				#print(encoder_example_test_variable.name, sess.run(encoder_example_test_variable))
				#print(decoder_example_test_variable.name, sess.run(decoder_example_test_variable))
				#print(encoder_example_test_variable.name, sess.run(encoder_example_test_variable))
				#print(decoder_example_test_variable.name, sess.run(decoder_example_test_variable))

				loss_val, reconstruction_loss_val, regularization_loss_val, reconstruction_energy_val, generation_energy_val, generator_reconstruction_energy_val, generator_generation_energy_val, introvae_generator_loss_val, introvae_inference_loss_val, _ = sess.run([loss, reconstruction_loss,
					regularization_loss, inference_reconstruction_energy, inference_generation_energy, generator_reconstruction_energy, generator_generation_energy,
					introvae_generator_loss, introvae_inference_loss, 
					inference_op], feed_dict=train_feed)
				sess.run(generator_op, feed_dict=train_feed)
				#print(encoder_example_test_variable.name, sess.run(encoder_example_test_variable))
				#print(decoder_example_test_variable.name, sess.run(decoder_example_test_variable))
				#exit()
				#test_val_a+=1
				#print("!%d"%test_val_a)
				#print()
				print_out = "step: %d,\ttotal loss: %.3f,\tKLD_losses: %.3f\t%.3f\t%.3f\t%.3f\t%.3f, reconstruction_loss: %.3f"%(step, loss_val, 
					regularization_loss_val, reconstruction_energy_val, generation_energy_val, generator_reconstruction_energy_val, generator_generation_energy_val, reconstruction_loss_val)
				#print_out = "step: %d,\ttotal loss: %.3f,\tKLD_losses: %.3f\t%.3f\t%.3f,\tGenerator loss: %.3f,\tInference loss: %.3f,\taverage stddev %s,\t stddev range %s\t%s,\taverage mean %s,\tmean range %s\t%s"%(
				#	step, loss_val, regularization_loss_val, reconstruction_energy_val, generation_energy_val, introvae_generator_loss_val, 
				#	introvae_inference_loss_val, *sess.run([*std_analysis, *mean_analysis], feed_dict=feed_dict))
				#print_out+=",\thyperparams:%.2f,%.2f,%.2f"%(alpha_multiplier_val, disc_offset_val, reconstruction_multiplier_val)
				print(print_out)
				#test_val_a+=1
				#print("!%d"%test_val_a)
				#print(""%sess.run([], feed_dict=feed_dict))
				log_file.write("%s\n"%print_out)
				if np.isnan(loss_val):
					break

				"""
				if not step%validation_step:
					######
					### Warning: Validation needs to be tuned for this.
					### TBD: Need to run the validation set above
					######
					train_feed[alpha_multiplier_val] = alpha_multiplier
					train_feed[disc_offset_val] = disc_offset
					train_feed[reconstruction_multiplier_val] = reconstruction_multiplier
					#test_val_a+=1
					#print("!%d"%test_val_a)
					validation_loss_val = sess.run(loss, feed_dict=validation_feed_dict)
					if prev_validation_loss is None or validation_loss_val < prev_validation_loss or ("validation_off" in kwargs and kwargs["validation_off"]):
						current_validation_count = 0
						#save model
						saver.save(sess, os.path.join(modelsavedir, "model.ckpt"))

						prev_validation_loss = validation_loss_val
					else:
						current_validation_count+=1
						if current_validation_count > validation_tolerence:
							break
				"""

			if is_train and (not step%plot_step or step in log_step):
				grid_size = [4, 6]
				grid_amount = reduce(lambda x,y: x*y, grid_size)
				if first_step:
					test_feed_dict = {  # keep all these images to test constant.
						inputs_ph: test_images[:grid_amount],
						outputs_ph: test_labels[:grid_amount],
					}
					first_step = False

				#create grid of interpolations between faces:
				#test_val_a+=1
				#print("!%d"%test_val_a)
				latent_space_generation = sess.run(latent_output, feed_dict=test_feed_dict)[:3]
				v1 = (latent_space_generation[1] - latent_space_generation[0])/(grid_size[0]-1)
				v2 = (latent_space_generation[2] - latent_space_generation[0])/(grid_size[1]-1)
				axis1 = np.arange(grid_size[0]).reshape(-1,1,1)*v1
				axis2 = np.arange(grid_size[1]).reshape(1,-1,1)*v2
				generation_latent_space = axis1+axis2+latent_space_generation[0]
				generation_latent_space = np.transpose(generation_latent_space, [1,0,2]).reshape(-1,latent_size)
				test_feed_dict[latents_ph] = generation_latent_space

				
				#save image of data:
				#create reconstruction
				#test_val_a+=1
				#print("!%d"%test_val_a)
				orig_images_val, recon_val, gener_val = sess.run([inputs[:grid_amount], reconstruction_prediction[:grid_amount], generation_prediction[:grid_amount]], feed_dict=test_feed_dict)
				
				#print("MIN, MAX", np.amin(recon_val), np.amax(recon_val))
				save_image_results = imgdir
				if "save_image_results" in kwargs:
					save_image_results = kwargs["save_image_results"]

				create_images_kwargs = {
					"original_images":orig_images_val,
					"reconstruction":recon_val,
					"generation":gener_val,
					}
				create_images(step, imgdir=save_image_results, postprocess_outputs=params["postprocess_outputs"], save_images_aspect_ratio=grid_size, **create_images_kwargs)
				saver.save(sess, os.path.join(modelsavedir, "model.ckpt"))
			
			if not is_train:
				test_feed_dict = {}
				create_images_kwargs = {}
				if not test_images is None:
					test_feed_dict[inputs_ph] = test_images
					orig_images_val, recon_val = sess.run([inputs, reconstruction_prediction], feed_dict=test_feed_dict)
					create_images_kwargs["original_images"] = orig_images_val[:48]
					create_images_kwargs["reconstruction"] = recon_val[:48]
					latent_analysis = sess.run(
						latent_analysis_package, feed_dict=test_feed_dict)
					#print()
					#print(modelsavedir[step].split("/")[1])
					return_dict[modelsavedir[step]] = latent_analysis
					#latent_analysis = np.transpose(latent_analysis)
					#for i in range(len(latent_analysis)):
					#	print("Latent element %d: mean, std, kld\t"%i, latent_analysis[i])

				if not test_latents is None:
					test_feed_dict[latents_ph] = test_latents
					gener_val = sess.run(generation_prediction, feed_dict=test_feed_dict)
					create_images_kwargs["generation"] = gener_val[:48]
				save_image_results = True
				if "save_image_results" in kwargs:
					save_image_results = kwargs["save_image_results"]
				save_images_aspect_ratio = None
				if "save_images_aspect_ratio" in kwargs:
					save_images_aspect_ratio = kwargs["save_images_aspect_ratio"]

				if not ("return_images" in kwargs and kwargs["return_images"]):
					create_images(step, imgdir=save_image_results, postprocess_outputs=params["postprocess_outputs"], save_images_aspect_ratio=save_images_aspect_ratio, **create_images_kwargs)
				else:
					pass


		if is_train:
			log_file.close()
		else:
			return return_dict

def create_images(step, imgdir, postprocess_outputs, save_images_aspect_ratio=None, original_images=None, reconstruction=None, generation=None):
	"""
	Creates 3 sets of images for VAE, reconstruction, generation, and original
	:param step: This is the current step of training. This will also be part of the name for the file.
	:param original_images: This is an array of original images, the ground truth
	:param reconstruction: This is the reconstructed images, from original_images.
	:param generation: This is the generated images,
	:return: None
	"""
	images_type = []
	possible_captions = ["reconstruction original images", "image reconstruction", "image generation"]
	captions = []
	j = 0
	for i in [original_images, reconstruction, generation]:
		if not i is None:
			height, width = gu.find_largest_factors(len(i))
			aspect_ratio = save_images_aspect_ratio if not save_images_aspect_ratio is None else [height, width]
			aspect_ratio = np.asarray(aspect_ratio).astype(int)
			#print(aspect_ratio)
			#exit()
			images_type.append(gu.create_image_grid(i, aspect_ratio))
			captions.append(possible_captions[j])
		j+=1
	# create images
	im = []
	header_size = 30  # amount of space for caption
	for i in range(len(images_type)):
		image_type = images_type[i]
		#image_type = np.log(5*(image_type+1))
		#image_type = image_type-np.amin(image_type)
		#image_type = image_type/np.amax(image_type)
		caption = captions[i]
		container = np.squeeze(np.ones((image_type.shape[0]+header_size, *image_type.shape[1:])))
		container[:-header_size] = image_type
		#print(np.uint8(postprocess_outputs(container)).dtype)
		im.append(Image.fromarray(np.uint8(postprocess_outputs(container))))
		#print("MIN, MAX", np.amin(container), np.amax(container))
		ImageDraw.Draw(im[i]).text((5,image_type.shape[0]+2), caption)

	width = max([i.size[0] for i in im])
	height = sum([i.size[1] for i in im])
	header = 40
	margin = 20

	if len(images_type[0].shape) == 3:
		n_channels = images_type[0].shape[2]
		total_image = Image.fromarray(np.ones((height+header, width+margin, n_channels), dtype=np.uint8)*255)
	else:
		total_image = Image.fromarray(np.ones((height+header, width+margin), dtype=np.uint8)*255)

	for i in range(len(im)):
		image = im[i]
		total_image.paste(image, (margin//2,header+i*image.size[1]))

	ImageDraw.Draw(total_image).text((margin//2+10,5), "Step: %d"%step)
	total_image.convert('RGB')
	if not imgdir is None:
		if not ".jpg" in imgdir:
			imgdir = os.path.join(imgdir, "image_%s.jpg"%step)
		total_image.save(imgdir)
	else:
		total_image.show()
		input()
if __name__ == "__main__":
	main()