from run_model import main
import general_utils as gu
import os
import numpy as np
import general_constants as gc

###########################
### Training The Models ###
###########################

#train with different KLD modifiers
#regular VAE
#main(is_train=True, save_folder="VAE", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
#	**kwargs: (0, 500, 1))
#IntroVAE
main(is_train=True, 
	save_folder="IntroVAE_test", 
	validation_off=True, 
	regularization_function=lambda step, disc_offset_val_past, 
		**kwargs: (
			0.15, # amount of detail (meat of the image)
			750, 
			1, # amount of structure and variance within the image.
			),
	load_prev=True)

#IntroVAE
#main(is_train=True, save_folder="IntroVAE_paper_params2", validation_off=True, #regularization_function=lambda step, disc_offset_val_past, 
#	**kwargs: (0.25, 750, 0.75))
"""
#IntroVAE with VAE pretrain
main(is_train=True, save_folder="IntroVAE_with_VAE_pretrain0", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
	**kwargs: (0 if step < 20000 else min(0.1, (step-20000)/10000), 500, 1))

#IntroVAE
main(is_train=True, save_folder="IntroVAE_0_1_alpha_250_m", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
	**kwargs: (0.1, 250, 1))

#IntroVAE
main(is_train=True, save_folder="IntroVAE_0_1_alpha_500_m", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
	**kwargs: (0.1, 500, 1))

#IntroVAE
main(is_train=True, save_folder="IntroVAE_0_1_alpha_500_m_0_5", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
	**kwargs: (0.1, 500, 0.5))

#IntroVAE
main(is_train=True, save_folder="IntroVAE_0_1_alpha_500_m_0_5", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
	**kwargs: (0.2, 500, 0.5))
"""
#IntroVAE
#main(is_train=True, save_folder="IntroVAE_0_25_alpha", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
#	**kwargs: (0.25, 500, 1))

#IntroVAE
#main(is_train=True, save_folder="IntroVAE_0_25_alpha_low_m", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
#	**kwargs: (0.25, 10, 1))

#IntroVAE
#main(is_train=True, save_folder="IntroVAE_0_25_alpha_high_m", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
#	**kwargs: (0.25, 5000, 1))

#IntroVAE with VAE pretrain
#main(is_train=True, save_folder="IntroVAE_with_VAE_pretrain1", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
#	**kwargs: (0 if step < 10000 else 0.25, 1000, min(step/50000,1)))

#IntroVAE with VAE pretrain
#main(is_train=True, save_folder="IntroVAE_with_VAE_pretrain2", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
#	**kwargs: (0 if step < 10000 else 0.25, 1000, 0.25 if step < 1000 else min(step/50000,0.75)+0.25))

#IntroVAE
#main(is_train=True, save_folder="IntroVAE_0_1_alpha", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
#	**kwargs: (0.1, 1000, 1))

#IntroVAE
#main(is_train=True, save_folder="IntroVAE_paper_params", validation_off=True, regularization_function=lambda step, disc_offset_val_past, 
#	**kwargs: (0.25, 200, 0.0125))
"""
###################################
### traversing the latent space ###
###################################

def create_interpolations(latent_space_anchor, latent_direction_1, latent_size, grid_size = 8):
	#create grid of interpolations between faces:
	v1 = (latent_direction_1 - latent_space_anchor)/(grid_size-1)
	axis1 = np.arange(grid_size).reshape(-1,1,1)*v1
	generation_latent_space = axis1+latent_space_anchor
	return generation_latent_space.reshape(-1,latent_size)

#run the models
dataset, get_group = gu.get_celeba_data(gc.datapath)#, preprocess_fn = lambda x: np.array(Image.fromarray(x).resize((x.shape[0], *imreshape_size, x.shape[-1]))))
test_images, test_labels = get_group(group_num=1, random_selection=True, remove_past=True)  # get images, and remove from get_group iterator
#test_images
#test_latents
#modelsavedir
logdir = "predictions"
modelsavedir = [os.path.join(logdir, i, "model_save_point") for i in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, i))]

return_dict = main(is_train=False, test_images=test_images, modelsavedir=modelsavedir, save_image_results=None, return_images=True)
images_path = []
for model, latent_data in return_dict.items():
	mean, std, kld, maximum, minimum = latent_data
	print()
	print(model)
	mean_sorted = np.argsort(np.abs(mean), axis=0).tolist()
	#kld_sorted = np.argsort(kld, axis=0).tolist()
	latent_space_anchor = np.zeros((32))
	latent_direction_1 = np.zeros((32))
	latent_space_anchor[mean_sorted[-1]] = maximum[mean_sorted[-1]]
	latent_direction_1[mean_sorted[-1]] = minimum[mean_sorted[-1]]
	num_images = 10
	latents = create_interpolations(latent_space_anchor, latent_direction_1, 32, num_images)
	latents = np.stack((latents, latents+std, latents+std*2))
	print(latents.shape)

	latents = latents.transpose(1,0,2).reshape(-1, 32)
	return_dict = main(is_train=False, test_latents=latents, modelsavedir=["/".join(model.split("/")[:-1])], 
		save_image_results=os.path.join("images", "%s.jpg"%model.split("/")[1]), return_images=False, save_images_aspect_ratio=[3,num_images])
	images_path.append(os.path.join("images", "%s.jpg"%model.split("/")[1]))
print(images_path)
#"""
##############################
### Disentanglement Metric ###
##############################
#TBD, this for more space to explore this.
