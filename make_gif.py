import imageio as iio 
import numpy as np 
import os
#creates the give from prediction images.

def make_gif(gif_path, images_path, use_images=True):
	#use_images: whether to use the images or create from the numpy saves
	def load_all_images(path, max_images = None, include_first=10):
		image_paths = [i for i in os.listdir(path) if i.endswith(".jpg")]
		if max_images:
			ipath_tmp = image_paths[:include_first]
			every_nth_image = (len(image_paths)-include_first)//max_images
			image_paths = ipath_tmp + [image_paths[i] for i in range(len(image_paths)) if i%every_nth_image]
		image_names = {int(i.split("_")[1][:-4]):os.path.join(path, i) for i in image_paths}
		image_index = list(image_names.keys())
		image_index.sort()
		image_names = [image_names[i] for i in image_index]
		images = [iio.imread(i) for i in image_names]
		return images

	if use_images:
		images = load_all_images(images_path)
	else:
		print("not implemented yet")
		exit()
	iio.mimsave(gif_path, images, duration=1/16)
