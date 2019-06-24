import numpy as np
import matplotlib.pyplot as plt
import sys
import os
image_num = int(sys.argv[1])

data = np.load("predictions/image_%d.npz"%image_num)
original_images = data["original_images"]
reconstruction = data["reconstruction"]
generation = data["generation"]
plt.clf()
subplot = [3,1]
f, axarr = plt.subplots(*subplot, constrained_layout=True)
axarr[0].imshow(original_images)
axarr[0].set_title("reconstruction original images")
axarr[1].imshow(reconstruction)
axarr[1].set_title("image reconstruction")
axarr[2].imshow(generation)
axarr[2].set_title("image generation")
f.savefig(os.path.join("./images", "image_%s.jpg"%image_num))
plt.close()